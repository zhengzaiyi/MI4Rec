import re
import os
import sys
import json
import pickle
import fsspec
import argparse
from tqdm import tqdm
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from accelerate import Accelerator

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config, AutoModel, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer, AutoTokenizer
from typing import List, Tuple
from collections import defaultdict
import torch.nn.functional as F

class EmbeddingMapper(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, num_classes):
        super(EmbeddingMapper, self).__init__()
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class UserItemMemory:
    '''
        Store {input_id: token} pairs for user and item tokens
        Store {input_id: metadata} pairs for user and item tokens
    '''
    def __init__(self):
        self.user_id2token = defaultdict(str)
        self.user_token2id = defaultdict(int)
        self.item_id2token = defaultdict(str)
        self.item_token2id = defaultdict(int)
        self.user_id2data = defaultdict(dict)
        self.item_id2data = defaultdict(dict)
        self.user_count = 0
        self.item_count = 0
    
    def update_user(self, input_id, token, data):
        if input_id in self.user_id2token:
            assert self.user_id2token[input_id] == token
            assert self.user_token2id[token] == input_id
        else:
            self.user_id2token[input_id] = token
            self.user_token2id[token] = input_id
        self.user_id2data[input_id].update(data)
        self.user_count = len(self.user_id2token)
    
    def update_item(self, input_id, token, data):
        if input_id in self.item_id2token:
            assert self.item_id2token[input_id] == token
            assert self.item_token2id[token] == input_id
        else:
            self.item_id2token[input_id] = token
            self.item_token2id[token] = input_id
        self.item_id2data[input_id].update(data)
        self.item_count = len(self.item_id2token)

def get_tokenizer_type(tokenizer_name):
    if "gpt2" in tokenizer_name.lower():
        return "BPE"
    elif "llama" in tokenizer_name.lower():
        return "sentencepiece"
    else:
        return None

class DPELLM4RecBaseModel(nn.Module):
    '''
        The base class for collaborative GPT model, i.e.,
        the GPT model with extra user/item embeddings
    '''
    def __init__(self, config, LLMmodel):
        super(DPELLM4RecBaseModel, self).__init__()
        # Obtain the number of users, items
        # and the size of the original vocabulary
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.config = config

        # Create new token embeddings for user/item tokens
        self.user_embeddings = nn.Embedding(self.num_users, config.n_embd)
        self.item_embeddings = nn.Embedding(self.num_items, config.n_embd)

        # Randomly initialize the new token embeddings
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
        # The pretrained gpt2 model
        self.LLMmodel = LLMmodel
        
    def embed(self, input_ids):
        # input_ids is a tensor of shape (batch_size, seq_length)
        vocab_mask = (input_ids < self.vocab_size).long() 
        user_mask = ((input_ids >= self.vocab_size) & (input_ids < self.vocab_size + self.num_users)).long() 
        item_mask = (input_ids >= self.vocab_size + self.num_users).long()
        
        # IDs outside of vocab range are set to 0
        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size-1)  
        vocab_embeddings = self.LLMmodel.wte(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)
        
        # IDs outside of user range are set to 0
        user_ids = ((input_ids - self.vocab_size) * user_mask).clamp_(0, self.num_users-1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)
        
        # IDs outside of item range are set to 0
        item_ids = ((input_ids - self.vocab_size - self.num_users) * item_mask).clamp_(0, self.num_items-1)
        item_embeddings = self.item_embeddings(item_ids)
        item_embeddings = item_embeddings * item_mask.unsqueeze(-1)

        # Sum up the embeddings as the input embeddings
        input_embeddings = vocab_embeddings + user_embeddings + item_embeddings
        return input_embeddings
        
    def forward(self, input_ids=None, **kwargs):
        # Obtain the embeddings of the input id sequence
        input_embeddings = self.embed(input_ids)
        # The input_embeds will be summed up with the pos_embed
        # And then forward into the transformer to get the results
        return self.LLMmodel(inputs_embeds=input_embeddings, **kwargs)

class DynamicDPELLM4RecBaseModel(nn.Module):
    '''
        The base class for collaborative GPT model, i.e.,
        the GPT model with extra user/item embeddings
    '''
    def __init__(
        self, 
        config, 
        LLMmodel, 
        memory: UserItemMemory,
        meta_logits_tokenizer: AutoTokenizer,
        meta_logits_classifier: AutoModelForSequenceClassification,
        # TODO: num_user_meta
        num_item_meta = 10,
        device = 'cuda',
        prob_norm = 'softmax',
        item_logits_infer = 'classifier',
        dataset_name = 'beauty'
    ):
        super(DynamicDPELLM4RecBaseModel, self).__init__()
        # Obtain the number of users, items
        # and the size of the original vocabulary
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.USET_OFFSET, self.ITEM_OFFSET = 0, 1
        self.config = config
        self.memory = memory
        self.device = device
        self.num_item_meta = num_item_meta
        self.item_logits_infer = item_logits_infer
        self.prob_norm = prob_norm

        # Create new token embeddings for user/item tokens
        self.user_embeddings = nn.Embedding(self.num_users, config.n_embd)
        self.meta_item_embeddings = nn.Embedding(self.num_item_meta, config.n_embd)
        self.all_item_ids = list(self.memory.item_id2data.keys())
        self.all_item_ids.sort()
        self.item_data = [' [SEP] '.join(self.memory.item_id2data[item_id].values()) for item_id in self.all_item_ids]
        # Randomly initialize the new token embeddings
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        import math
        self.meta_item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
        # The pretrained gpt2 model
        self.LLMmodel = LLMmodel

        if item_logits_infer == 'original':
            self.item_embeddings = nn.Embedding(self.num_items, config.n_embd).to(device)
            self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
            del self.meta_item_embeddings
        elif item_logits_infer == 'stella':
            all_item_src_embs = pickle.load(open(os.path.join('/shared/user/embs', f"{dataset_name}_item_review_embeddings.pkl"), 'rb'))
            self.item_emb_mapper = EmbeddingMapper(all_item_src_embs.shape[1], [400], self.num_item_meta).to(device) 
            self.item_src_embs = nn.Embedding(all_item_src_embs.shape[0], all_item_src_embs.shape[1]).to(device)
            self.item_src_embs.weight.data.copy_(torch.from_numpy(all_item_src_embs))
            self.item_src_embs.weight.requires_grad = False
        elif item_logits_infer in ['classifier', 'bert']:
            self.meta_logits_tokenizer = meta_logits_tokenizer
            self.meta_logits_classifier = meta_logits_classifier.to(device)
            if item_logits_infer == 'classifier':
                for name, param in self.meta_logits_classifier.named_parameters():
                    if 'classifier' not in name:
                        param.requires_grad = False
            self.all_item_hidden_states = self.get_all_item_hidden_states()
            self.all_item_hidden_states = self.all_item_hidden_states.detach()
        elif item_logits_infer == 'direct' or item_logits_infer == 'random':
            self.item_logits = nn.Embedding(self.num_items, num_item_meta)
            if item_logits_infer == 'random':
                self.item_logits.weight.requires_grad = False

    def get_probs(self, logits):
        if self.prob_norm == 'softmax':
            return F.softmax(logits, dim=1)
        elif self.prob_norm == 'sigmoid':
            return F.sigmoid(logits)
        else:
            return logits
    
    def _add_single_item(self, item_data):
        self.item_data.append(item_data)
        new_item_data_tokenized = self.meta_logits_tokenizer(item_data, padding=True, truncation=True, return_tensors="pt")
        new_item_hidden_states = self.meta_logits_classifier(**new_item_data_tokenized).last_hidden_state
        self.all_item_hidden_states = torch.cat([self.all_item_hidden_states, new_item_hidden_states.pooler_output], dim=0)
        return new_item_hidden_states

    def get_all_item_hidden_states(self):
        batch_size = 64
        all_hidden_states = []
        # from tqdm import tqdm
        for i in range(0, len(self.item_data), batch_size):
            interval = slice(i, i+batch_size) if i+batch_size < len(self.item_data) else slice(i, len(self.item_data))
            item_data_tokenized = self.meta_logits_tokenizer(
                self.item_data[interval], 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt",
                max_length=90
            )
            item_data_tokenized = {k: v.to(self.device) for k, v in item_data_tokenized.items()}
            all_hidden_states.append(self.meta_logits_classifier.bert(**item_data_tokenized).pooler_output)
        return torch.cat(all_hidden_states, dim=0)

    def embed(self, input_ids):
        # input_ids is a tensor of shape (batch_size, seq_length)
        vocab_mask = (input_ids < self.vocab_size).long() 
        user_mask = ((input_ids >= self.vocab_size) & ((input_ids - self.vocab_size) % 2 == self.USET_OFFSET)).long() 
        item_mask = ((input_ids >= self.vocab_size) & ((input_ids - self.vocab_size) % 2 == self.ITEM_OFFSET)).long() 
        
        # IDs outside of vocab range are set to 0
        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size-1)  
        vocab_embeddings = self.LLMmodel.wte(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)
        
        # IDs outside of user range are set to 0
        user_ids = (((input_ids - self.vocab_size) * user_mask) // 2).clamp_(0, self.memory.user_count - 1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)
        
        # IDs outside of item range are set to 0
        item_ids = (((input_ids - self.vocab_size) * item_mask) // 2).clamp_(0, self.memory.item_count - 1)
        # item_embeddings = self.item_embeddings(item_ids)
        if self.item_logits_infer == 'original':
            weighted_item_embeddings = self.item_embeddings(item_ids)
            weighted_item_embeddings = weighted_item_embeddings * item_mask.unsqueeze(-1)
        elif self.item_logits_infer in ['classifier', 'bert']:
            item_logits = self.meta_logits_classifier.classifier(self.all_item_hidden_states[item_ids])
            item_probs = self.get_probs(item_logits)
            weighted_item_embeddings = torch.matmul(item_probs, self.meta_item_embeddings.weight)
            weighted_item_embeddings = weighted_item_embeddings * item_mask.unsqueeze(-1)
        elif self.item_logits_infer in ['stella']:
            item_src_embs = self.item_src_embs(item_ids)
            item_logits = self.item_emb_mapper(item_src_embs)
            item_probs = self.get_probs(item_logits)
            weighted_item_embeddings = torch.matmul(item_probs, self.meta_item_embeddings.weight)
            weighted_item_embeddings = weighted_item_embeddings * item_mask.unsqueeze(-1)
        elif self.item_logits_infer in ['woMI']:
            item_src_embs = self.item_src_embs(item_ids)
            weighted_item_embeddings = self.item_emb_mapper(item_src_embs)
            weighted_item_embeddings = weighted_item_embeddings * item_mask.unsqueeze(-1)
        elif self.item_logits_infer == 'direct' or self.item_logits_infer == 'random':
            item_logits = self.item_logits(item_ids)
            item_probs = self.get_probs(item_logits)
            weighted_item_embeddings = torch.matmul(item_probs, self.meta_item_embeddings.weight)
            weighted_item_embeddings = weighted_item_embeddings * item_mask.unsqueeze(-1)

        # Sum up the embeddings as the input embeddings
        input_embeddings = vocab_embeddings + user_embeddings + weighted_item_embeddings
        return input_embeddings
        
    def forward(self, input_ids=None, **kwargs):
        # Obtain the embeddings of the input id sequence
        input_embeddings = self.embed(input_ids)
        # The input_embeds will be summed up with the pos_embed
        # And then forward into the transformer to get the results
        return self.LLMmodel(inputs_embeds=input_embeddings, **kwargs)

class MSEDynamicDPELLM4RecBaseModel(DynamicDPELLM4RecBaseModel):
    '''
        The base class for collaborative GPT model, i.e.,
        the GPT model with extra user/item embeddings
    '''
    def __init__(
        self, 
        config, 
        LLMmodel, 
        memory: UserItemMemory,
        meta_logits_tokenizer: AutoTokenizer,
        meta_logits_classifier: AutoModelForSequenceClassification,
        # TODO: num_user_meta
        num_item_meta = 10,
        device = 'cuda',
        prob_norm = 'softmax',
        item_logits_infer = 'classifier',
        dataset_name = 'beauty'
    ):
        super(MSEDynamicDPELLM4RecBaseModel, self).__init__(config, LLMmodel, memory, meta_logits_tokenizer, meta_logits_classifier, num_item_meta, device, prob_norm, item_logits_infer, dataset_name)
        # Obtain the number of users, items
        # and the size of the original vocabulary
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.USET_OFFSET, self.ITEM_OFFSET = 0, 1
        self.config = config
        self.memory = memory
        self.device = device
        self.num_item_meta = num_item_meta
        self.item_logits_infer = item_logits_infer
        self.prob_norm = prob_norm

        # Create new token embeddings for user/item tokens
        self.user_embeddings = nn.Embedding(self.num_users, config.n_embd)
        self.meta_item_embeddings = nn.Embedding(self.num_item_meta, config.n_embd)
        self.all_item_ids = list(self.memory.item_id2data.keys())
        self.all_item_ids.sort()
        self.item_data = [' [SEP] '.join(self.memory.item_id2data[item_id].values()) for item_id in self.all_item_ids]
        # Randomly initialize the new token embeddings
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        import math
        self.meta_item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
        # The pretrained gpt2 model
        self.LLMmodel = LLMmodel

        # if item_logits_infer == 'original':
        #     self.item_embeddings = nn.Embedding(self.num_items, config.n_embd).to(device)
        #     self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        assert item_logits_infer in ['classifier', 'bert', 'stella', 'direct', 'random', 'woMI', 'original']
        self.item_embeddings = nn.Embedding(self.num_items, config.n_embd).to(device)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if item_logits_infer == 'stella':
            all_item_src_embs = pickle.load(open(os.path.join('/shared/user/embs', f"{dataset_name}_item_review_embeddings.pkl"), 'rb'))
            self.item_emb_mapper = EmbeddingMapper(all_item_src_embs.shape[1], [400], self.num_item_meta).to(device) 
            self.item_src_embs = nn.Embedding(all_item_src_embs.shape[0], all_item_src_embs.shape[1]).to(device)
            self.item_src_embs.weight.data.copy_(torch.from_numpy(all_item_src_embs))
            self.item_src_embs.weight.requires_grad = False
        if item_logits_infer == 'woMI':
            all_item_src_embs = pickle.load(open(os.path.join('/shared/user/embs', f"{dataset_name}_item_review_embeddings.pkl"), 'rb'))
            self.item_emb_mapper = EmbeddingMapper(all_item_src_embs.shape[1], [400], config.n_embd).to(device) 
            self.item_src_embs = nn.Embedding(all_item_src_embs.shape[0], all_item_src_embs.shape[1]).to(device)
            self.item_src_embs.weight.data.copy_(torch.from_numpy(all_item_src_embs))
            # self.meta_item_embeddings = nn.Embedding(all_item_src_embs.shape[1], config.n_embd)
            del self.meta_item_embeddings
            # self.item_src_embs.weight.requires_grad = False
            # self.meta_item_embeddings = nn.Embedding(config.n_embd, config.n_embd)
            # self.meta_item_embeddings.weight.data.copy_(torch.eye(config.n_embd))
            # self.meta_item_embeddings.weight.requires_grad = False
        elif item_logits_infer in ['classifier', 'bert']:
            self.meta_logits_tokenizer = meta_logits_tokenizer
            self.meta_logits_classifier = meta_logits_classifier.to(device)
            if item_logits_infer == 'classifier':
                for name, param in self.meta_logits_classifier.named_parameters():
                    if 'classifier' not in name:
                        param.requires_grad = False
            self.all_item_hidden_states = self.get_all_item_hidden_states()
            self.all_item_hidden_states = self.all_item_hidden_states.detach()
        elif item_logits_infer == 'direct' or item_logits_infer == 'random':
            self.item_logits = nn.Embedding(self.num_items, num_item_meta)
            if item_logits_infer == 'random':
                self.item_logits.weight.requires_grad = False
                
    def original_embed(self, input_ids):
        # input_ids is a tensor of shape (batch_size, seq_length)
        vocab_mask = (input_ids < self.vocab_size).long() 
        user_mask = ((input_ids >= self.vocab_size) & ((input_ids - self.vocab_size) % 2 == self.USET_OFFSET)).long() 
        item_mask = ((input_ids >= self.vocab_size) & ((input_ids - self.vocab_size) % 2 == self.ITEM_OFFSET)).long() 
        
        # IDs outside of vocab range are set to 0
        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size-1)  
        vocab_embeddings = self.LLMmodel.wte(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)
        
        # IDs outside of user range are set to 0
        user_ids = (((input_ids - self.vocab_size) * user_mask) // 2).clamp_(0, self.memory.user_count - 1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)
        
        # IDs outside of item range are set to 0
        item_ids = (((input_ids - self.vocab_size) * item_mask) // 2).clamp_(0, self.memory.item_count - 1)
        # item_embeddings = self.item_embeddings(item_ids)
        weighted_item_embeddings = self.item_embeddings(item_ids)
        weighted_item_embeddings = weighted_item_embeddings * item_mask.unsqueeze(-1)

        # Sum up the embeddings as the input embeddings
        input_embeddings = vocab_embeddings + user_embeddings + weighted_item_embeddings
        return input_embeddings

class ContentGPTForUserItemWithLMHeadBatch(nn.Module):
    '''
        This class conducts language modeling to learn both
        user/item token embeddings via textual data, where
        we view the texts that include user/item ID as prompt.
        E.g.,
            inputs_ids_prompt:
              "user_1 writes the following review for item_1:"
            inputs_ids_main:
              "This item is too expensive."
        where we only calculate LM loss on the main texts.
    '''
    def __init__(self, config, base_model):
        super(ContentGPTForUserItemWithLMHeadBatch, self).__init__()
        self.base_model = base_model
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between the output layer and the token embeddings
        self.lm_head.weight = self.base_model.LLMmodel.wte.weight

    def forward(self, 
                input_ids_prompt, 
                input_ids_main, 
                labels_main=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                collaborative_embeds=None,
                **kwargs):
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, 
                                         return_dict=True, **kwargs)
        past_key_values = outputs_prompt.past_key_values

        # Calculate the language modeling loss for the main texts
        outputs_main = self.base_model(input_ids=input_ids_main, 
                                       past_key_values=past_key_values, 
                                       attention_mask=attention_mask,
                                       return_dict=True)

        lm_logits = self.lm_head(outputs_main.last_hidden_state)
        outputs = (lm_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            
            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]
            
            active_loss = attention_mask[:, prompt_length+1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the token sequences
            loss = loss_fct(active_logits, active_labels)
            
            # Mutual regularization loss
            if regularize:
                # User/Item token embeddings only appear in the prompt
                content_embeds = self.base_model.embed(input_ids_prompt)
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        content_embeds,
                        collaborative_embeds)
                )
                loss += regularize_loss
                outputs = (loss, regularize_loss) + outputs            
            else:
                outputs = (loss,) + outputs
        return outputs

class CollaborativeGPTwithItemLMHeadBatch(nn.Module):
    '''
        Collaborative filtering model to learn user/item embeddings.
    '''
    def __init__(self, config, base_model):
        super(CollaborativeGPTwithItemLMHeadBatch, self).__init__()

        # Obtain the number of users, items, and vocabulary size
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model
        
        # Item recommendation head
        self.item_head = nn.Linear(config.n_embd, self.num_items, bias=False)
        
        # Tie the weights between the item embeddings and the item recommendation head
        self.item_head.weight = self.base_model.item_embeddings.weight 

    def forward(self,
                input_ids_prompt,
                input_ids_main,
                labels_main=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                content_embeds=None,
                **kwargs):
        # Base model forward pass for the prompt text
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, 
                                         return_dict=True, 
                                         **kwargs)
        past_key_values = outputs_prompt.past_key_values

        # Base model forward pass for the main text with attention mask
        outputs_main = self.base_model(input_ids=input_ids_main,
                                       past_key_values=past_key_values,
                                       attention_mask=attention_mask,
                                       return_dict=True)

        item_logits = self.item_head(outputs_main.last_hidden_state)
        outputs = (item_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = item_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            shift_labels = shift_labels - self.vocab_size - self.num_users

            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]
        
            active_loss = attention_mask[:, prompt_length+1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the item sequences
            loss = loss_fct(active_logits, active_labels)
            
            # Mutual regularization loss
            if regularize:
                collaborative_embeds = torch.cat(
                    (self.base_model.embed(input_ids_prompt),
                     self.base_model.embed(input_ids_main)),
                    axis=1
                )
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        collaborative_embeds,
                        content_embeds)
                )
                loss += regularize_loss
                outputs = (loss, regularize_loss) + outputs
            else:
                outputs = (loss,) + outputs
        return outputs

class DynamicCollaborativeGPTwithItemLMHeadBatch(nn.Module):
    '''
        Collaborative filtering model to learn user/item embeddings.
    '''
    def __init__(
        self, 
        config, 
        base_model: DynamicDPELLM4RecBaseModel,
        device = 'cuda'
    ):
        super(DynamicCollaborativeGPTwithItemLMHeadBatch, self).__init__()

        # Obtain the number of users, items, and vocabulary size
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model
        
        # Item recommendation head
        # weighted sum of meta embeddings, Tie the weights between the item embeddings and the item recommendation head
        self.device = device
        
        
    def _add_single_item(self, new_item_hidden_states):
        # abandoned
        self.all_item_logits = torch.cat([self.all_item_logits, new_item_hidden_states], dim=0)
        self.item_head.to(self.device)
        self.item_head = torch.matmul(self.all_item_logits, self.base_model.meta_item_embeddings.weight)

    def get_item_head(self,):
        if self.base_model.item_logits_infer == 'original':
            return self.base_model.item_embeddings.weight
        elif self.base_model.item_logits_infer in ['classifier', 'bert']:
            all_item_logits = self.base_model.meta_logits_classifier.classifier(
                self.base_model.all_item_hidden_states
            )
        elif self.base_model.item_logits_infer in ['stella', 'woMI']:
            all_item_logits = self.base_model.item_emb_mapper(self.base_model.item_src_embs.weight)
        elif self.base_model.item_logits_infer == 'direct' or self.base_model.item_logits_infer == 'random':
            all_item_logits = self.base_model.item_logits.weight
        if self.base_model.item_logits_infer == 'woMI':
            return all_item_logits
        all_item_probs = self.base_model.get_probs(all_item_logits)
        item_head = torch.matmul(all_item_probs, self.base_model.meta_item_embeddings.weight).to(self.device)
        return item_head

    def forward(self,
                input_ids_prompt,
                input_ids_main,
                labels_main=None,
                attention_mask=None,
                regularize=False,
                orthogonal=False,
                lambda_V=None,
                content_embeds=None,
                **kwargs):
        # Base model forward pass for the prompt text
        item_head = self.get_item_head()
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, 
                                         return_dict=True, 
                                         **kwargs)
        past_key_values = outputs_prompt.past_key_values

        # Base model forward pass for the main text with attention mask
        outputs_main = self.base_model(input_ids=input_ids_main,
                                       past_key_values=past_key_values,
                                       attention_mask=attention_mask,
                                       return_dict=True)
        item_logits = torch.matmul(outputs_main.last_hidden_state, item_head.T)
        outputs = (item_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = item_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            shift_labels = (shift_labels - self.vocab_size) // 2

            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]
        
            active_loss = attention_mask[:, prompt_length+1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the item sequences
            all_losses = [loss_fct(active_logits, active_labels)]
            
            # Mutual regularization loss
            if regularize:
                collaborative_embeds = torch.cat(
                    (self.base_model.embed(input_ids_prompt),
                     self.base_model.embed(input_ids_main)),
                    axis=1
                )
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        collaborative_embeds,
                        content_embeds)
                )
                all_losses[0] += regularize_loss
                all_losses.append(regularize_loss)
                
            if orthogonal:
                # Orthogonal regularization loss
                meta_item_embeds = self.base_model.meta_item_embeddings.weight
                ortho_loss = lambda_V * torch.mean(
                    torch.matmul(meta_item_embeds, meta_item_embeds.T).pow(2)
                )
                all_losses[0] += ortho_loss
                all_losses.append(ortho_loss)
            
            outputs = tuple(all_losses) + outputs
        return outputs

class CollaborativeGPTwithItemRecommendHead(nn.Module):
    '''
        Recommend items to a user according to input queries.
        multinomial likelihood is put on all the items for a user.
    '''
    def __init__(self, config, base_model):
        super(CollaborativeGPTwithItemRecommendHead, self).__init__()
        # Obtain the number of users and items
        self.num_users = config.num_users
        self.num_items = config.num_items

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model
        
        # Item recommendation head
        self.item_head = nn.Linear(config.n_embd, self.num_items, bias=False)
        
        # Tie the weights between the item embeddings and the item recommendation head
        self.item_head.weight = self.base_model.item_embeddings.weight 

    def forward(self, 
                input_ids=None, 
                target_ids=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                main_ids=None,
                content_embeds=None,
                **kwargs):    
        transformer_outputs = self.base_model(input_ids, 
                                              attention_mask=attention_mask, 
                                              **kwargs)
        hidden_states = transformer_outputs[0]

        # Find the indices of the last non-padding tokens
        last_non_pad_token_indices = attention_mask.sum(dim=1) - 1

        # Gather the last non-padding token embeddings
        last_token_hidden_states = torch.stack([
            hidden_states[i, idx, :] for i, idx in \
                enumerate(last_non_pad_token_indices)
        ])

        # Calculate the item scores
        item_scores = self.item_head(last_token_hidden_states)

        # Convert scores to multinomial probabilities
        item_log_probs = F.log_softmax(item_scores, dim=-1)
        
        # Calculating the multinomial loss
        neg_ll = -torch.mean(torch.sum(item_log_probs * target_ids, dim=-1))
        
        if regularize:
            # User/Item token embeddings only appear in the prompt
            rec_embeds_prompt = self.base_model.embed(input_ids)
            rec_embeds_target = self.base_model.embed(main_ids)
            rec_embeds = torch.cat(
                (rec_embeds_prompt, rec_embeds_target),
                axis=1
            )
            regularize_loss = lambda_V * torch.mean(
                nn.MSELoss(reduction='sum')(
                    rec_embeds,
                    content_embeds)
            )
            neg_ll += regularize_loss
            outputs = (neg_ll, regularize_loss, item_log_probs)
        else: 
            outputs = (neg_ll, item_log_probs)
        return outputs

class DynamicCollaborativeGPTwithItemRecommendHead(nn.Module):
    '''
        Recommend items to a user according to input queries.
        multinomial likelihood is put on all the items for a user.
    '''
    def __init__(
        self, 
        config, 
        base_model: DynamicDPELLM4RecBaseModel,
        device = 'cuda'
    ):
        super(DynamicCollaborativeGPTwithItemRecommendHead, self).__init__()

        # Obtain the number of users, items, and vocabulary size
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model
        
        # Item recommendation head
        # weighted sum of meta embeddings, Tie the weights between the item embeddings and the item recommendation head
        self.device = device

    def get_item_head(self,):
        if self.base_model.item_logits_infer == 'original':
            return self.base_model.item_embeddings.weight
        elif self.base_model.item_logits_infer in ['classifier', 'bert']:
            all_item_logits = self.base_model.meta_logits_classifier.classifier(
                self.base_model.all_item_hidden_states
            )
        elif self.base_model.item_logits_infer == 'direct' or self.base_model.item_logits_infer == 'random':
            all_item_logits = self.base_model.item_logits.weight
        elif self.base_model.item_logits_infer in ['stella', 'woMI']:
            all_item_logits = self.base_model.item_emb_mapper(self.base_model.item_src_embs.weight)
        if self.base_model.item_logits_infer == 'woMI':
            return all_item_logits
        all_item_probs = self.base_model.get_probs(all_item_logits)
        item_head = torch.matmul(all_item_probs, self.base_model.meta_item_embeddings.weight).to(self.device)
        return item_head

    def forward(self, 
                input_ids=None, 
                target_ids=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                main_ids=None,
                content_embeds=None,
                warm_item_idx=None,
                **kwargs):    
        item_head = self.get_item_head()
        original_item_head = self.base_model.item_embeddings.weight
        transformer_outputs = self.base_model(input_ids, 
                                              attention_mask=attention_mask, 
                                              **kwargs)
        hidden_states = transformer_outputs[0]

        # Find the indices of the last non-padding tokens
        last_non_pad_token_indices = attention_mask.sum(dim=1) - 1
        # TODO: check hidden_states
        # Gather the last non-padding token embeddings
        last_token_hidden_states = torch.stack([
            hidden_states[i, idx, :] for i, idx in \
                enumerate(last_non_pad_token_indices)
        ])

        # Calculate the item scores
        item_scores = torch.matmul(last_token_hidden_states, item_head.T)
        original_item_scores = torch.matmul(last_token_hidden_states, original_item_head.T)
        # Convert scores to multinomial probabilities
        item_log_probs = F.log_softmax(item_scores, dim=-1)
        original_item_log_probs = F.log_softmax(original_item_scores, dim=-1)
        # Calculating the multinomial loss
        neg_ll = -torch.mean(torch.sum(item_log_probs * target_ids, dim=-1))
        neg_ll = neg_ll - torch.mean(torch.sum(original_item_log_probs * target_ids, dim=-1)) * (1 - lambda_V)# TODO: check this
        if regularize:
            # User/Item token embeddings only appear in the prompt
            rec_embeds_prompt = self.base_model.embed(input_ids)
            rec_embeds_target = self.base_model.embed(main_ids)
            rec_embeds = torch.cat(
                (rec_embeds_prompt, rec_embeds_target),
                axis=1
            )
            if isinstance(self.base_model, MSEDynamicDPELLM4RecBaseModel):
                original_rec_embeds_prompt = self.base_model.original_embed(input_ids)
                original_rec_embeds_target = self.base_model.original_embed(main_ids)
                content_embeds = torch.cat(
                    (original_rec_embeds_prompt, original_rec_embeds_target),
                    axis=1
                )
            regularize_loss = lambda_V * torch.mean(
                nn.MSELoss(reduction='sum')(
                    rec_embeds,
                    content_embeds
                )
            )
            neg_ll += regularize_loss
            outputs = (neg_ll, regularize_loss, item_log_probs)
        else: 
            outputs = (neg_ll, item_log_probs)
        return outputs


    
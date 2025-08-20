import re
import os
import sys
import json
import pickle
import fsspec
import argparse
from tqdm import tqdm

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
from transformers import GPT2Tokenizer, AutoTokenizer, PreTrainedTokenizerFast
from typing import List, Tuple
from collections import defaultdict
import torch.nn.functional as F
from model import UserItemMemory
import wget



class BPETokenizer(AutoTokenizer):
    def __init__(self, 
                 vocab_file, 
                 merges_file, 
                 num_users,
                 num_items,
                 **kwargs):
        super().__init__(vocab_file=vocab_file, 
                         merges_file=merges_file, 
                         **kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.user_token_encoder = self._add_user_token_encoder()
        self.item_token_encoder = self._add_item_token_encoder()
        
        #We add the user/item token encoders to the original vocab encoder
        self.encoder.update(self.user_token_encoder)
        self.encoder.update(self.item_token_encoder)
        
        #We add the corresponding decoders to the original vocab decoder
        self.user_token_decoder = {v:k for k,v in self.user_token_encoder.items()}
        self.item_token_decoder = {v:k for k,v in self.item_token_encoder.items()}
        self.decoder.update(self.user_token_decoder)
        self.decoder.update(self.item_token_decoder)
        
    
    def _add_user_token_encoder(self):
        return {"user_{}".format(i):(i+self.vocab_size) 
                for i in range(self.num_users)}
    
    def _add_item_token_encoder(self):
        return {"item_{}".format(j):(j+self.vocab_size+self.num_users)
                for j in range(self.num_items)}
    
    def _pre_tokenize(self, text):
        '''
            In this function, we break down the sentence that 
            describes user/item features or their historical 
            interactions into pieces, where the ID word like
            user_i or item_j is kept as a single piece. 
            
            E.g.,
                text = "This is user_1's comment about item_3 
                        after he bought the item"
                pieces = ['This is', 'user_1', "'s comment about", 
                          'item_3', ' after he bought the item']
                          
            Note that we keep the space on the left of a word to 
            show that the word does not appear on the beginning 
            part of a sentence.
        '''
        pattern = r'(user_\d+|item_\d+)'
        matches = re.findall(pattern, text)
        pieces = re.split(pattern, text)
        pieces = [piece.rstrip() for piece in pieces if piece.rstrip()]
        return pieces
    
    def _tokenize(self, text):
        '''
            Please note that when the token is a user/item token,
            we don't distinguish whether it appears on the start
            of the a sentence or not.
        '''
        split_tokens = []
        pieces = self._pre_tokenize(text)
        for piece in pieces:
            # If piece is a user ID
            # piece is itself a token
            if piece in self.user_token_encoder.keys():
                split_tokens.append(piece)
            # If piece is an item ID
            # piece is also a token
            elif piece in self.item_token_encoder.keys():
                split_tokens.append(piece)
            # If piece is a sentence
            # Use the original tokenization to
            # further break down piece
            else:
                split_tokens += super()._tokenize(piece)
        return split_tokens

class BPETokenizerBatch(BPETokenizer):
    """
     tokenizer class that extends TokenizerWithUserItemIDTokens
     and supports batch encoding.
    """
    def __init__(self, vocab_file, merges_file, num_users, num_items, **kwargs):
        super().__init__(vocab_file=vocab_file, merges_file=merges_file,
                         num_users=num_users, num_items=num_items, **kwargs)
        # Set the padding token ID to 0
        self.pad_token_id = 0
    
    def encode_batch(self, texts, max_length=None):
        """
        Encodes a batch of texts into input IDs and attention masks.

        Args:
            texts (List[str]): List of input texts to be encoded.
            max_length (int, optional): Maximum length of the encoded 
                sequences. If None, the maximum length in the batch 
                will be used. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the 
                input IDs and attention masks as NumPy arrays.
        """
        encoded_inputs = []
        max_length_batch = max(len(self._tokenize(text)) for text in texts)
        
        # Determine the maximum length for padding
        if (not max_length) or max_length <= max_length_batch:
            max_length = max_length_batch
        
        for text in texts:
            tokens = self._tokenize(text)
            input_ids = self.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Pad the sequence to the max_length
            padding_length = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            
            encoded_inputs.append((input_ids, attention_mask))
        
        input_ids_batch, attention_mask_batch = zip(*encoded_inputs)
        return np.array(input_ids_batch), np.array(attention_mask_batch)

class DynamicBPETokenizer(GPT2Tokenizer):
    def __init__(self, 
                 vocab_file, 
                 merges_file, 
                 num_users,
                 num_items,
                 memory: UserItemMemory,
                 user_files: List[str] = None,
                 item_files: List[str] = None,
                 **kwargs):
        super().__init__(vocab_file=vocab_file, 
                         merges_file=merges_file, 
                         **kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.memory = memory

        self.USER_OFFSET, self.ITEM_OFFSET = 0, 1
        self._init_user_tokens()
        self._init_item_tokens()
        # self.read_files(user_files, item_files)
        
        #We add the user/item token encoders to the original vocab encoder
        self.encoder.update(self.memory.user_token2id)
        self.encoder.update(self.memory.item_token2id)
        
        #We add the corresponding decoders to the original vocab decoder
        self.decoder.update(self.memory.user_id2token)
        self.decoder.update(self.memory.item_id2token)

    def read_files(self, user_files, item_files):
        user_pattern = r'user_\d+'
        item_pattern = r'item_\d+'
        for user_file in user_files:
            with fsspec.open(user_file, "r") as f:
                for line in f:
                    user_token = re.search(user_pattern, line).group()
                    if user_token not in self.memory.user_token2id:
                        continue
                    user_id = self.memory.user_token2id[user_token]
                    self.memory.user_id2data[user_id].update({
                        user_file.split("/")[-1].split(".")[0]: line
                    })
        for item_file in item_files:
            if '.txt' in item_file:
                with fsspec.open(item_file, "r") as f:
                    for line in f:
                        item_token = re.search(item_pattern, line).group()
                        if item_token not in self.memory.item_token2id:
                            continue
                        item_id = self.memory.item_token2id[item_token]
                        self.memory.item_id2data[item_id].update({
                            item_file.split("/")[-1].split(".")[0]: line
                        })
            elif '.pkl' in item_file:
                with fsspec.open(item_file, "rb") as f:
                    item_datas = pickle.load(f)
                    for item_data in item_datas:
                        text = ' '.join(item_data)
                        item_token = re.search(item_pattern, text).group()
                        if item_token not in self.memory.item_token2id:
                            continue
                        item_id = self.memory.item_token2id[item_token]
                        self.memory.item_id2data[item_id].update({
                            item_file.split("/")[-1].split(".")[0]: text
                        })
                    
    
    def _init_user_tokens(self):
        for i in range(self.num_users):
            token = f"user_{i}"
            self.memory.update_user(
                2 * i + self.vocab_size + self.USER_OFFSET, 
                token, 
                {}
            )
    
    def _init_item_tokens(self):
        for i in range(self.num_items):
            token = f"item_{i}"
            self.memory.update_item(
                2 * i + self.vocab_size + self.ITEM_OFFSET, 
                token, 
                {}
            )

    def _add_single_user(self, user_token):
        assert user_token not in self.memory.user_token2id
        user_id = 2 * self.memory.user_count + self.vocab_size + self.USER_OFFSET
        self.encoder[user_token] = user_id
        self.decoder[user_id] = user_token
        return user_id
    
    def _add_single_item(self, item_token, item_data):
        assert item_token not in self.memory.item_token2id
        item_id = 2 * self.memory.item_count + self.vocab_size + self.ITEM_OFFSET
        self.encoder[item_token] = item_id
        self.decoder[item_id] = item_token
        return item_id
    
    def _pre_tokenize(self, text):
        '''
            In this function, we break down the sentence that 
            describes user/item features or their historical 
            interactions into pieces, where the ID word like
            user_i or item_j is kept as a single piece. 
            
            E.g.,
                text = "This is user_1's comment about item_3 
                        after he bought the item"
                pieces = ['This is', 'user_1', "'s comment about", 
                          'item_3', ' after he bought the item']
                          
            Note that we keep the space on the left of a word to 
            show that the word does not appear on the beginning 
            part of a sentence.
        '''
        pattern = r'(user_\d+|item_\d+)'
        matches = re.findall(pattern, text)
        pieces = re.split(pattern, text)
        pieces = [piece.rstrip() for piece in pieces if piece.rstrip()]
        return pieces
    
    def _tokenize(self, text):
        '''
            Please note that when the token is a user/item token,
            we don't distinguish whether it appears on the start
            of the a sentence or not.
        '''
        split_tokens = []
        pieces = self._pre_tokenize(text)
        for piece in pieces:
            # If piece is a user ID
            # piece is itself a token
            if piece in self.memory.user_token2id.keys():
                split_tokens.append(piece)
            # If piece is an item ID
            # piece is also a token
            elif piece in self.memory.item_token2id.keys():
                split_tokens.append(piece)
            # If piece is a sentence
            # Use the original tokenization to
            # further break down piece
            else:
                split_tokens += super()._tokenize(piece)
        return split_tokens

class DynamicBPETokenizerBatch(DynamicBPETokenizer):
    """
     tokenizer class that extends TokenizerWithUserItemIDTokens
     and supports batch encoding.
    """
    def __init__(
        self, 
        vocab_file,
        merges_file, 
        num_users, 
        num_items, 
        memory,
        user_files: List[str] = None,
        item_files: List[str] = None,
        **kwargs):
        super().__init__(vocab_file=vocab_file, merges_file=merges_file,
                         num_users=num_users, num_items=num_items, memory=memory, user_files=user_files, item_files=item_files, **kwargs)
        # Set the padding token ID to 0
        self.pad_token_id = 0
    
    def encode_batch(self, texts, max_length=None):
        """
        Encodes a batch of texts into input IDs and attention masks.

        Args:
            texts (List[str]): List of input texts to be encoded.
            max_length (int, optional): Maximum length of the encoded 
                sequences. If None, the maximum length in the batch 
                will be used. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the 
                input IDs and attention masks as NumPy arrays.
        """
        encoded_inputs = []
        max_length_batch = max(len(self._tokenize(text)) for text in texts)
        
        # Determine the maximum length for padding
        if (not max_length) or max_length <= max_length_batch:
            max_length = max_length_batch
        
        for text in texts:
            tokens = self._tokenize(text)
            input_ids = self.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Pad the sequence to the max_length
            padding_length = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            
            encoded_inputs.append((input_ids, attention_mask))
        
        input_ids_batch, attention_mask_batch = zip(*encoded_inputs)
        return np.array(input_ids_batch), np.array(attention_mask_batch)

# sentencepiece
class DynamicSPTokenizer(PreTrainedTokenizerFast):
    def __init__(self, 
                 spm_model_path, # tokenizer.json
                 num_users,
                 num_items,
                 memory: UserItemMemory,
                 user_files: List[str] = None,
                 item_files: List[str] = None,
                 **kwargs):
        super().__init__(tokenizer_file=spm_model_path, **kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.memory = memory

        self.USER_OFFSET, self.ITEM_OFFSET = 0, 1
        self._init_user_tokens()
        self._init_item_tokens()
        # self.read_files(user_files, item_files)
        
        #We add the user/item token encoders to the original vocab encoder

    def read_files(self, user_files, item_files):
        user_pattern = r'user_\d+'
        item_pattern = r'item_\d+'
        for user_file in user_files:
            with fsspec.open(user_file, "r") as f:
                for line in f:
                    user_token = re.search(user_pattern, line).group()
                    if user_token not in self.memory.user_token2id:
                        continue
                    user_id = self.memory.user_token2id[user_token]
                    self.memory.user_id2data[user_id].update({
                        user_file.split("/")[-1].split(".")[0]: line
                    })
        for item_file in item_files:
            with fsspec.open(item_file, "r") as f:
                for line in f:
                    item_token = re.search(item_pattern, line).group()
                    if item_token not in self.memory.item_token2id:
                        continue
                    item_id = self.memory.item_token2id[item_token]
                    self.memory.item_id2data[item_id].update({
                        item_file.split("/")[-1].split(".")[0]: line
                    })
                    
    def _init_user_tokens(self):
        for i in range(self.num_users):
            token = f"user_{i}"
            id_ = 2 * i + len(self.vocab) + self.USER_OFFSET
            self.memory.update_user(
                id_,
                token, 
                {}
            )
    
    def _init_item_tokens(self):
        for i in range(self.num_items):
            token = f"item_{i}"
            id_ = 2 * i + len(self.vocab) + self.ITEM_OFFSET
            self.memory.update_item(
                id_,
                token, 
                {}
            )
            
    def _pre_tokenize(self, text):
        '''
            In this function, we break down the sentence that 
            describes user/item features or their historical 
            interactions into pieces, where the ID word like
            user_i or item_j is kept as a single piece. 
            
            E.g.,
                text = "This is user_1's comment about item_3 
                        after he bought the item"
                pieces = ['This is', 'user_1', "'s comment about", 
                          'item_3', ' after he bought the item']
                          
            Note that we keep the space on the left of a word to 
            show that the word does not appear on the beginning 
            part of a sentence.
        '''
        pattern = r'(user_\d+|item_\d+)'
        matches = re.findall(pattern, text)
        pieces = re.split(pattern, text)
        pieces = [piece.rstrip() for piece in pieces if piece.rstrip()]
        return pieces
    
    def _tokenize(self, text):
        '''
            Please note that when the token is a user/item token,
            we don't distinguish whether it appears on the start
            of the a sentence or not.
        '''
        split_tokens = []
        pieces = self._pre_tokenize(text)
        for piece in pieces:
            # If piece is a user ID
            # piece is itself a token
            if piece in self.memory.user_token2id.keys():
                split_tokens.append(piece)
            # If piece is an item ID
            # piece is also a token
            elif piece in self.memory.item_token2id.keys():
                split_tokens.append(piece)
            # If piece is a sentence
            # Use the original tokenization to
            # further break down piece
            else:
                split_tokens += super()._tokenize(piece)
        return split_tokens

class DynamicSPTokenizerBatch(DynamicSPTokenizer):
    """
     tokenizer class that extends TokenizerWithUserItemIDTokens
     and supports batch encoding.
    """
    def __init__(
        self, 
        spm_model_path,
        num_users, 
        num_items, 
        memory,
        user_files: List[str] = None,
        item_files: List[str] = None,
        **kwargs):
        super().__init__(spm_model_path=spm_model_path,
                         num_users=num_users, num_items=num_items, memory=memory, user_files=user_files, item_files=item_files, **kwargs)
        # Set the padding token ID to 0
        self.pad_token_id = 0
    
    def encode_batch(self, texts, max_length=None):
        """
        Encodes a batch of texts into input IDs and attention masks.

        Args:
            texts (List[str]): List of input texts to be encoded.
            max_length (int, optional): Maximum length of the encoded 
                sequences. If None, the maximum length in the batch 
                will be used. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the 
                input IDs and attention masks as NumPy arrays.
        """
        encoded_inputs = []
        max_length_batch = max(len(self._tokenize(text)) for text in texts)
        
        # Determine the maximum length for padding
        if (not max_length) or max_length <= max_length_batch:
            max_length = max_length_batch
        
        for text in texts:
            tokens = self._tokenize(text)
            input_ids = self.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Pad the sequence to the max_length
            padding_length = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            
            encoded_inputs.append((input_ids, attention_mask))
        
        input_ids_batch, attention_mask_batch = zip(*encoded_inputs)
        return np.array(input_ids_batch), np.array(attention_mask_batch)


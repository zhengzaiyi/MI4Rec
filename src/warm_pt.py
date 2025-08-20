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
from scipy.sparse import csr_matrix, issparse, lil_matrix
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config, AutoModel, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer, AutoTokenizer
from typing import List, Tuple
from collections import defaultdict
from model import (
    UserItemMemory,
    DynamicDPELLM4RecBaseModel,
    ContentGPTForUserItemWithLMHeadBatch,
    DynamicCollaborativeGPTwithItemLMHeadBatch,
)
from tokenizer import DynamicBPETokenizerBatch, DynamicSPTokenizerBatch
from data import (
    UserItemContentGPTDatasetBatch,
    CollaborativeGPTGeneratorBatch,
)
from typing import Dict
def add_single_item(
        tokenizer: DynamicBPETokenizerBatch, 
        memory: UserItemMemory, 
        base_model: DynamicDPELLM4RecBaseModel, 
        collab_model: DynamicCollaborativeGPTwithItemLMHeadBatch,
        item_token: str, 
        item_data: Dict[str, str],
    ):
    '''
        Add a single item to the tokenizer, memory, base_model, and collab_model.
        Memery is shared among all the models.
    '''
    if item_token in memory.item_token2id:
        return
    item_id = tokenizer._add_single_item(item_token, item_data)
    memory.update_item(item_id, item_token, item_data)
    new_item_hidden_states = base_model._add_single_item(
        ' [SEP] '.join(item_data.values())
    )
    collab_model._add_single_item(new_item_hidden_states)
    # TODO: Update

   
    
def save_local(remote_path, local_path, remote_mode, local_mode):
    '''
        Save the remote file in remote_path
        to the local_path...
    '''
    with fsspec.open(remote_path, remote_mode) as f:
        content = f.read()
    with fsspec.open(local_path, local_mode) as f:
        f.write(content)


def save_local_folder(remote_path, local_path, remote_mode="rb", local_mode="wb", fs=None):
    '''
        Recursively save all files from the remote folder at remote_path
        to the local folder at local_path, preserving the folder structure.
    '''
    assert fs is not None, "Please provide the filesystem object!"
    def recursive_copy(remote_dir, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
        for item in fs.ls(remote_dir):
            if fs.isdir(item):
                sub_remote_dir = item
                sub_local_dir = os.path.join(local_dir, os.path.basename(item))
                recursive_copy(sub_remote_dir, sub_local_dir)
            else:
                local_file_path = os.path.join(local_dir, os.path.basename(item))
                with fs.open(item, remote_mode) as remote_file:
                    content = remote_file.read()
                with open(local_file_path, local_mode) as local_file:
                    local_file.write(content)

    recursive_copy(remote_path, local_path)

def save_remote(local_path, remote_path, local_mode, remote_mode):
    '''
        Save the local file in local_path
        to the remote_path...
    '''
    with fsspec.open(local_path, local_mode) as f:
        content = f.read()
    with fsspec.open(remote_path, remote_mode) as f:
        f.write(content)

def save_folder_to_fsspec(local_folder_path, remote_folder_path, fs):
    
    for root, _, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            remote_file_path = os.path.join(remote_folder_path, relative_path)

            with open(local_file_path, 'rb') as local_file:
                with fsspec.open(remote_file_path, 'wb') as remote_file:
                    remote_file.write(local_file.read())

def mask_columns(matrix, mask_ratio, type='random', seed=100):
    if seed is not None:
        np.random.seed(seed)
    
    num_cols = matrix.shape[1]
    num_mask_cols = int(num_cols * mask_ratio)
    
    mask_indices = np.random.choice(num_cols, num_mask_cols, replace=False)
    
    if issparse(matrix):
        masked_matrix = matrix.copy().tolil()
        masked_matrix[:, mask_indices] = 0
        masked_matrix = masked_matrix.tocsr()
        row_nonzero = masked_matrix.getnnz(axis=1)
        non_zero_row_indices = row_nonzero != 0
        masked_matrix = masked_matrix[non_zero_row_indices]
    else:
        masked_matrix = matrix.copy()
        masked_matrix[:, mask_indices] = 0
        row_nonzero = np.count_nonzero(masked_matrix, axis=1)
        masked_matrix = masked_matrix[row_nonzero != 0]
    
    return masked_matrix, non_zero_row_indices, mask_indices

def main():
    # Define the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--model", type=str,
        help="specify the size of the model")
    parser.add_argument("--batch_size", type=int,
        help="specify the batch size")
    parser.add_argument("--num_meta", type=int,
        help="number of meta_embeddings")
    parser.add_argument('--shared_root', type=str, default="",)
    parser.add_argument('--model_root', type=str, default="",)
    parser.add_argument('--local_root', type=str, default="tmp",)
    parser.add_argument('-c', '--cold_start', type=float, default=0.0,)
    parser.add_argument('--item_logits_infer', type=str, default="classifier",)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,)
    parser.add_argument('-p', '--prob_norm', type=str, default="softmax",)
    parser.add_argument('-cp', '--cold_start_type', type=str, default="random",)
    args = parser.parse_args()
    path_suffix = f"{args.lambda_V}_{args.num_meta}_{args.cold_start}_{args.item_logits_infer}_{args.learning_rate}"
    from time import sleep
    # sleep(20000)

    shared_root = args.shared_root
    model_root = args.model_root
    local_root = args.local_root
    dataset = args.dataset
    model = args.model
    lambda_V = float(args.lambda_V)
    batch_size = args.batch_size
    num_meta = args.num_meta
    model_path = f'/shared/public/models/'
    learning_rate = args.learning_rate
    prob_norm = args.prob_norm

    cold_start_suffix = f"_{args.cold_start}" if args.cold_start > 0 else ""
    # cold_start_suffix = ""
    if not os.path.exists(local_root):
        os.makedirs(local_root, exist_ok=True)
    
    assert dataset in ["beauty", "toys", "sports", "yelp", "company"]
    assert model in ["gpt2", "llama2-7b", "llama3.2-1B"]

    host = "hdfs://ltx1-holdem//"
    port = 8443
    fs = fsspec.filesystem("file")
    
    accelerator.print("-----Current Setting-----")
    accelerator.print(f"cuda: {torch.cuda.is_available()}")
    accelerator.print(f"dataset: {dataset}")
    accelerator.print(f"model: {model}")
    accelerator.print(f"lambda_V: {args.lambda_V}")
    accelerator.print(f"batch_size: {args.batch_size}")
    accelerator.print(f"num_meta: {num_meta}")
    accelerator.print(f"item_logits_infer: {args.item_logits_infer}")
    accelerator.print(f"cold start rate: {args.cold_start}")
    
    # Define the number of GPUs to be used
    num_gpus = torch.cuda.device_count()
    accelerator.print(f"num_gpus: {num_gpus}")
    
    '''
        Get the basic information of the dataset
    '''
    accelerator.print("-----Begin Obtaining Dataset Info-----")
    data_root = os.path.join(shared_root, "dataset", dataset)
    
    # meta_path = os.path.join(data_root, f"meta{cold_start_suffix}.pkl")
    # train_mat_path = os.path.join(data_root, f"train_matrix{cold_start_suffix}.npz")
    
    # train_mat = load_npz(train_mat_path)
    # train_mat, non_zero_row_indices, mask_indices= mask_columns(train_mat, args.cold_start, type=args.cold_start_type)
    other_cold_suffix = f'/{round(1-args.cold_start, 2)}' if args.cold_start > 0 and args.cold_start != 0.2 else ""
    train_mat_path = os.path.join(
        f'{data_root}{other_cold_suffix}', 
        f"{'warm_' if args.cold_start > 0 else ''}train_matrix.npz"
    )
    train_mat = load_npz(train_mat_path)
    accelerator.print(f"Loading data from {train_mat_path}...")
    

    # with fsspec.open(meta_path, "rb") as f:
    #     meta_data = pickle.load(f)
        
    num_users = train_mat.shape[0]
    num_items = train_mat.shape[1]
    accelerator.print(f"num_users: {num_users}")
    accelerator.print(f"num_items: {num_items}")
    accelerator.print("-----End Obtaining Dataset Info-----\n")


    '''
        Obtain the tokenizer with user/item tokens
    '''
    accelerator.print("-----Begin Obtaining the Tokenizer-----")
    tokenizer_root = os.path.join(shared_root, "model", "pretrained", model)
    accelerator.print(f"Loading pretrained tokenizer from {tokenizer_root}...")
    vocab_file = os.path.join(tokenizer_root, "vocab.json")
    merges_file = os.path.join(tokenizer_root, "merges.txt")
    spe_file = os.path.join(local_root, "tokenizer.json")

    bert_tokenizer_path = os.path.join(shared_root, "model", "pretrained", "bert_tokenizer")
    bert_model_path = os.path.join(shared_root, "model", "pretrained", "bert_model")

    accelerator.wait_for_everyone()
    memory = UserItemMemory()
    from data import ITEM_CONTENT_FILES
    if 'gpt2' in model.lower():
        tokenizer = DynamicBPETokenizerBatch(vocab_file,
            merges_file,
            num_users,
            num_items,
            memory,
            [],
            [os.path.join(data_root, "item_texts", file) for file in ITEM_CONTENT_FILES]
        )
    elif 'llama' in model.lower():
        # TODO: from pretrained
        tokenizer = DynamicSPTokenizerBatch(vocab_file,
            merges_file,
            num_users,
            num_items,
            memory,
            [],
            [os.path.join(data_root, "item_texts", file) for file in ITEM_CONTENT_FILES]
        )
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")
    collaborative_data_gen = CollaborativeGPTGeneratorBatch(tokenizer, train_mat)

    '''
        Define the review data generator
    '''
    accelerator.print("-----Begin Obtaining the Review Data Generator-----")
    review_path = os.path.join(data_root, "user_item_texts", f"review.pkl")
    accelerator.print(f"Loading data from {review_path}...")
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, review_path)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Review Data Generator-----\n")


    '''
        Now we deal with the user/item interaction data
    '''
    accelerator.print("-----Begin Obtaining the Collaborative Data Generator-----")
    
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Collaborative Data Generator-----\n")


    '''
        Extend the config of the original GPT model
    '''
    accelerator.print("-----Begin Setting Up the Config-----")
    # Update the config
    _config_path =  os.path.join(model_path, model, "config.json")
    with fsspec.open(_config_path, "r") as f:
        _config = json.load(f)
    if 'gpt2' in model.lower():  
        config = GPT2Config(**_config)
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(_config_path)
        if hasattr(config, "hidden_size"):
            config.n_embd = config.hidden_size
    config.num_users = num_users
    config.num_items = num_items
    accelerator.print("Success!")
    accelerator.print("-----End Setting Up the Config-----\n")


    '''
        Instantiate the pretrained GPT2 model
    '''
    accelerator.print("-----Begin Instantiating the Pretrained GPT Model-----")
    if 'gpt2' in model.lower():
        LLMmodel = GPT2Model(config)
    elif 'llama' in model.lower():
        LLMmodel = AutoModel.from_pretrained(config)
    pretrained_root = os.path.join(shared_root, "model", "pretrained")
    accelerator.print(f"Loading pretrained weights from {pretrained_root}...")
    local_pretrained_weights_path = os.path.join(pretrained_root, model, "pytorch_model.bin")
    local_llm_folder = os.path.join(pretrained_root, model)
    if 'gpt2' in model.lower():
        LLMmodel.load_state_dict(torch.load(local_pretrained_weights_path), strict=False)
    elif 'llama' in model.lower():
        LLMmodel.from_pretrained(local_llm_folder)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pretrained GPT Model-----\n")


    '''
        Instantiate the GPT for recommendation content model
    '''
    accelerator.print("-----Begin Instantiating the Content GPT Model-----")
    # content_base_model = DPELLM4RecBaseModel(config, LLMmodel)
    content_base_model = DynamicDPELLM4RecBaseModel(
        config, 
        LLMmodel, 
        memory, 
        AutoTokenizer.from_pretrained(bert_tokenizer_path), # TODO: use local tokenizer
        AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_meta, ignore_mismatched_sizes=True), # TODO: use local classifier
        device=device,
        num_item_meta=args.num_meta,
        item_logits_infer='original',
        prob_norm=prob_norm,
    )
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Content GPT Model-----\n")


    '''
        Freeze the parameters of the pretrained GPT2 for content model
    '''
    for name, param in content_model.named_parameters():
        # we allow only user/item token embeddings to be trained
        if ('user' not in name) and \
           ('item' not in name) and \
           ('classifier' not in name):
            param.requires_grad = False

    accelerator.print("-----Trainable Parameters-----")
    for name, param in content_model.named_parameters():
        if param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))
    
    # accelerator.print("\n-----Non-trainable Parameters-----")
    # for name, param in content_model.named_parameters():
    #     if not param.requires_grad:
            # accelerator.print("{} : {}".format(name, param.shape))


    '''
        Instantiate the GPT for recommendation collaborative model
    '''
    accelerator.print("-----Begin Instantiating the Collaborative GPT Model-----")
    # collaborative_base_model = DPELLM4RecBaseModel(config, LLMmodel)
    collaborative_base_model = DynamicDPELLM4RecBaseModel(
        config, 
        LLMmodel, 
        memory, 
        AutoTokenizer.from_pretrained(bert_tokenizer_path), # TODO: use local tokenizer
        AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_meta, ignore_mismatched_sizes=True), 
        device=device,
        num_item_meta=args.num_meta,
        item_logits_infer='original',
        prob_norm=prob_norm,
    )
    collaborative_model = DynamicCollaborativeGPTwithItemLMHeadBatch(config, collaborative_base_model, device=device)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Collaborative GPT Model-----\n")


    '''
        Freeze the parameters of the pretrained GPT2 for collaborative model
    '''
    for name, param in collaborative_model.named_parameters():
        # we allow only user/item token embeddings to be trained
        if ('user' not in name) and \
           ('item' not in name) and \
           ('classifier' not in name):
            param.requires_grad = False

    accelerator.print("-----Trainable Parameters-----")
    for name, param in collaborative_model.named_parameters():
        if param.requires_grad:
            print("{} : {}".format(name, param.shape))
        
    # accelerator.print("\n-----Non-Trainable Parameters-----")
    # for name, param in collaborative_model.named_parameters():
    #     if not param.requires_grad:
    #         accelerator.print("{} : {}".format(name, param.shape))


    '''
        Set up the training details
    '''
    accelerator.print("-----Begin Setting Up the Training Details-----")
    # learning_rate = 1e-3
    num_pretrained_epochs = 5
    num_epochs = 30

    '''
        Create a data sampler for distributed training
    '''
    accelerator.print("-----Begin Creating the DataLoader-----")
    # Create the review data loader with the custom collate_fn
    review_data_loader = DataLoader(review_data_gen, 
                                    batch_size=batch_size, 
                                    collate_fn=review_data_gen.collate_fn)

    # Create the collaborative data loader with the custon collate_fn
    collaborative_data_loader = DataLoader(collaborative_data_gen, 
                                           batch_size=batch_size, 
                                           collate_fn=collaborative_data_gen.collate_fn)
    accelerator.print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    content_model.train()
    content_model.to(device)

    collaborative_model.train()
    collaborative_model.to(device)

    # Obtain the optimizer
    review_optimizer = optim.Adam(content_model.parameters(), 
                                  lr=1e-3)

    collaborative_optimizer = optim.Adam(collaborative_model.parameters(), 
                                         lr=learning_rate)
    # collab_scheduler = optim.lr_scheduler.StepLR(collaborative_optimizer, step_size=10, gamma=0.1)
    
    # Parallel model, optimizer and data loader with accelerator
    content_model, review_optimizer, review_data_loader = accelerator.prepare(
        content_model, review_optimizer, review_data_loader
    )

    # Parallel model, optimizer and data loader with accelerator
    collaborative_model, collaborative_optimizer, collaborative_data_loader = accelerator.prepare(
        collaborative_model, collaborative_optimizer, collaborative_data_loader
    )

    # Initialize best_loss with infinity
    review_best_loss = float('inf')
    collaborative_best_loss = float('inf')

    # The place to save the content model weights
    content_model_root = os.path.join(model_root, "model", dataset, model, "content")
    if not fs.exists(content_model_root):
        fs.makedirs(content_model_root, exist_ok=True)
    accelerator.print(f"Weights will be saved to {content_model_root}!")
    
    # The place to save the collaborative model weights
    collaborative_model_root = os.path.join(model_root, "model", dataset, model, "collab")
    if not fs.exists(collaborative_model_root):
        fs.makedirs(collaborative_model_root, exist_ok=True)
    accelerator.print(f"Weights will be saved to {collaborative_model_root}!")

    accelerator.print("-----End Setting Up the Training Details-----\n")

    '''
        Define the pretraining loop for the content GPT
    '''
    accelerator.print("-----Begin Content GPT Pretraining Loop-----")
    for epoch in range(num_pretrained_epochs):
        review_total_loss = 0
        
        # Initialize tqdm progress bar
        # progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}", 
        #                     disable=not accelerator.is_local_main_process, ncols=80)
        for input_ids_prompt, input_ids_main, attention_mask in review_data_loader:
            review_optimizer.zero_grad()

            # Obtain the data
            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = content_model(input_ids_prompt, 
                                    input_ids_main, 
                                    labels_main=input_ids_main,
                                    attention_mask=attention_mask)
            review_loss = outputs[0]

            # Backward pass and optimization
            accelerator.backward(review_loss)
            review_optimizer.step()

            review_total_loss += review_loss.item()
            # progress_bar.set_postfix({"Review Loss": review_loss.item()})

        thread_review_average_loss = torch.tensor([review_total_loss / len(review_data_loader)]).to(device)
        gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
        review_average_loss = torch.mean(gathered_review_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")

        # Check if the current loss is better than the best_loss
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings in the main process 
            content_weight_root = os.path.join(content_model_root, f"content_model_{path_suffix}.pth")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(content_model).state_dict(), content_weight_root)
    accelerator.print("-----End Content GPT Pretraining Loop-----")


    '''
        Iteratively training the collaborative and content GPT model for recommendations
    '''
    accelerator.print("-----Begin the Iterative Training Loop-----")
    for epoch in range(num_epochs):
        '''
            Optimize the collaborative GPT model
        '''
        collaborative_total_loss = 0
        regularize_total_loss = 0
        
        # progress_bar = tqdm(collaborative_data_loader, desc=f"Epoch {epoch + 1}",
        #                     disable=not accelerator.is_local_main_process, ncols=100)
        for input_ids_prompt, input_ids_main, attention_mask in collaborative_data_loader:
            collaborative_optimizer.zero_grad()

            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                content_embeds = torch.cat(
                    (accelerator.unwrap_model(content_model).base_model.embed(input_ids_prompt),
                     accelerator.unwrap_model(content_model).base_model.embed(input_ids_main)),
                    axis=1
                ).to(device)
                
            # Forward pass of the collaborative GPT
            outputs = collaborative_model(input_ids_prompt, 
                                          input_ids_main, 
                                          labels_main=input_ids_main,
                                          attention_mask=attention_mask,
                                          regularize=True,
                                          lambda_V=lambda_V,
                                          content_embeds=content_embeds)
            collaborative_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            accelerator.backward(collaborative_loss)
            collaborative_optimizer.step()
            
            collaborative_total_loss += collaborative_loss.item()
            regularize_total_loss += regularize_loss.item()
            
            # progress_bar.set_postfix({"Collaborative Loss": collaborative_loss.item(),
            #                           "Regularize Loss": regularize_loss.item()})
        
        # Gather the collaborative LM loss from different device
        thread_collaborative_average_loss = torch.tensor([collaborative_total_loss / len(collaborative_data_loader)]).to(device)
        gathered_collaborative_average_loss = accelerator.gather(thread_collaborative_average_loss)
        collaborative_average_loss = torch.mean(gathered_collaborative_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Collaborative Loss: {collaborative_average_loss:.4f}")
        
        # Gather the regularize loss from difference device
        thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(collaborative_data_loader)]).to(device)
        gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
        regularize_average_loss = torch.mean(gathered_regularize_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")
        
        # Check if the current loss is better than the best_loss
        if collaborative_average_loss < collaborative_best_loss:
            collaborative_best_loss = collaborative_average_loss

            # Save user embeddings in the main process
            collab_weight_root = os.path.join(collaborative_model_root, f"collab_model_{path_suffix}.pth")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(collaborative_model).state_dict(), collab_weight_root)

        
        '''
            Optimize the content GPT model
        '''
        review_total_loss = 0
        regularize_total_loss = 0
        
        # progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}", 
        #                     disable=not accelerator.is_local_main_process, ncols=100)
        for input_ids_prompt, input_ids_main, attention_mask in review_data_loader:
            review_optimizer.zero_grad()

            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                collaborative_embeds = accelerator.unwrap_model(collaborative_model).\
                                       base_model.embed(input_ids_prompt).to(device)
                
            # Forward pass of the content GPT
            outputs = content_model(input_ids_prompt, 
                                    input_ids_main, 
                                    labels_main=input_ids_main,
                                    attention_mask=attention_mask,
                                    regularize=True,
                                    lambda_V=lambda_V,
                                    collaborative_embeds=collaborative_embeds)
            review_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            accelerator.backward(review_loss)
            review_optimizer.step()

            review_total_loss += review_loss.item()
            regularize_total_loss += regularize_loss.item()
            # progress_bar.set_postfix({"Review Loss": review_loss.item(),
            #                           "Regularize Loss": regularize_loss.item()})

        # Gather the content LM loss from different device
        thread_review_average_loss = torch.tensor([review_total_loss / len(review_data_loader)]).to(device)
        gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
        review_average_loss = torch.mean(gathered_review_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")
        
        # Gather the regularize loss from different device
        thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(review_data_loader)]).to(device)
        gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
        regularize_average_loss = torch.mean(gathered_regularize_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Check if the current loss is better than the best_loss
        accelerator.wait_for_everyone()
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings in the main process
            content_weight_root = os.path.join(content_model_root, f"content_model_{path_suffix}.pth")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(content_model).state_dict(), content_weight_root)


if __name__ == "__main__":
    main()
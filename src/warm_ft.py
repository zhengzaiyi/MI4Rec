import re
import os
import sys
import json
import pickle
import fsspec
import random
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from accelerate import Accelerator

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config, AutoModel
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from model import (
    UserItemMemory,
    DynamicDPELLM4RecBaseModel,
    MSEDynamicDPELLM4RecBaseModel,
    ContentGPTForUserItemWithLMHeadBatch,
    DynamicCollaborativeGPTwithItemRecommendHead,
)
from tokenizer import DynamicBPETokenizerBatch
from data import (
    UserItemContentGPTDatasetBatch,
    RecommendationGPTTestGeneratorBatch,
    RecommendationGPTTrainGeneratorBatch,
)
def Recall_at_k(y_true, y_pred, k, agg="sum"):
    '''
        Average recall for top k recommended results.
        The training records should be set to -inf in y_pred
    '''
    batch_size = y_pred.shape[0]
    topk_idxes = np.argpartition(-y_pred, k, axis=1)[:, :k]
    y_pred_bin = np.zeros_like(y_pred, dtype=bool)
    y_pred_bin[np.arange(batch_size)[:, None], topk_idxes] = True
    y_true_bin = (y_true > 0)
    true_positives_count = np.sum(y_true_bin, axis=1)
    valid_mask = true_positives_count > 0
    hits = np.sum(np.logical_and(y_true_bin, y_pred_bin), axis=-1).astype(np.float32)
    hits = hits[valid_mask]
    # print(np.sum(y_true_bin, axis=1), hits)
    recalls = hits/np.minimum(k, np.sum(y_true_bin, axis=1)[valid_mask])
    if agg == "sum":
        recall = np.sum(recalls)
    elif agg == "mean":
        recall = np.mean(recalls)
    else:
        raise NotImplementedError(f"aggregation method {agg} not defined!")
    return recall


def NDCG_at_k(y_true, y_pred, k, agg="sum"):
    assert not np.isnan(y_true).any(), "y_true contains NaN values"
    assert not np.isnan(y_pred).any(), "y_pred contains NaN values"
    assert k <= y_pred.shape[1], f"k={k} exceeds prediction dimension {y_pred.shape[1]}"

    batch_size = y_pred.shape[0]
    topk_idxes_unsort = np.argpartition(-y_pred, k, axis=1)[:, :k]
    topk_value_unsort = y_pred[np.arange(batch_size)[:, None],topk_idxes_unsort]
    topk_idxes_rel = np.argsort(-topk_value_unsort, axis=1)
    topk_idxes = topk_idxes_unsort[np.arange(batch_size)[:, None], topk_idxes_rel]
    y_true_topk = y_true[np.arange(batch_size)[:, None], topk_idxes]
    y_true_bin = (y_true > 0).astype(np.float32)
    weights = 1./np.log2(np.arange(2, k + 2))
    DCG = np.sum(y_true_topk*weights, axis=-1)
    true_positives_count = np.sum(y_true_bin, axis=1)
    normalizer = np.array([np.sum(weights[:int(n)]) for n in np.minimum(k, true_positives_count)])
    valid_mask = true_positives_count > 0
    DCG = DCG[valid_mask]
    normalizer = normalizer[valid_mask]
    non_zero_mask = normalizer > 0
    DCG = DCG[non_zero_mask]
    normalizer = normalizer[non_zero_mask]
    NDCG_scores = DCG / normalizer
    if agg == "sum":
        NDCG = np.sum(NDCG_scores)
    elif agg == "mean":
        NDCG = np.mean(NDCG_scores) if NDCG_scores.size > 0 else 0.0
    else:
        raise NotImplementedError(f"aggregation method {agg} not defined!")
    return NDCG
  
def save_local(remote_path, local_path, remote_mode, local_mode):
    '''
        Save the remote file in remote_path
        to the local_path...
    '''
    with fsspec.open(remote_path, remote_mode) as f:
        content = f.read()
    with fsspec.open(local_path, local_mode) as f:
        f.write(content)


def save_remote(local_path, remote_path, local_mode, remote_mode):
    '''
        Save the local file in local_path
        to the remote_path...
    '''
    with fsspec.open(local_path, local_mode) as f:
        content = f.read()
    with fsspec.open(remote_path, remote_mode) as f:
        f.write(content)


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
    parser.add_argument('-cp', '--cold_start_type', type=str, default="random",)
    parser.add_argument('--item_logits_infer', type=str, default="classifier",)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,)
    parser.add_argument('--pt_lr', type=float, default=1e-2,)
    parser.add_argument('--pt_lv', type=float, default=0.1,)
    parser.add_argument('-p', '--prob_norm', type=str, default="softmax",)
    parser.add_argument('--skip_pt', action='store_true',)
    args = parser.parse_args()
    load_path_suffix = f"{args.pt_lv}_{300}_{args.cold_start}_{'stella'}_{args.pt_lr}"
    path_suffix = f"{args.lambda_V}_{args.num_meta}_{args.cold_start}_{args.item_logits_infer}_{args.learning_rate}"
    # from time import sleep
    # sleep(20000)
    
    shared_root = args.shared_root
    model_root = args.model_root
    local_root = args.local_root
    dataset = args.dataset
    lambda_V = float(args.lambda_V)
    batch_size = args.batch_size
    num_meta = args.num_meta
    model = args.model
    model_path = f'/shared/public/models/'
    cold_start_suffix = ""
    learning_rate = args.learning_rate
    prob_norm = args.prob_norm
    if not os.path.exists(local_root):
        os.makedirs(local_root, exist_ok=True)
    
    assert dataset in ["beauty", "toys", "sports", "yelp", "company"]

    
    assert model in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    

    host = "hdfs://ltx1-holdem//"
    port = 8443
    fs = fsspec.filesystem('file')
    
    accelerator.print("-----Current Setting-----")
    accelerator.print(f"dataset: {dataset}")
    accelerator.print(f"model: {model}")
    accelerator.print(f"lambda_V: {args.lambda_V}")
    accelerator.print(f"batch_size: {args.batch_size}")
    accelerator.print(f"cold start rate: {args.cold_start}")
    # Define the number of GPUs to be used
    num_gpus = torch.cuda.device_count()
    accelerator.print(f"num_gpus: {num_gpus}")
    
    '''
        Get the basic information of the dataset
    '''
    accelerator.print("-----Begin Obtaining Dataset Info-----")
    data_root = os.path.join(shared_root, "dataset", dataset)
    '''
        Obtain the training/validation data generator
    '''
    accelerator.print("-----Begin Obtaining the Collaborative Data Generator-----")
    other_cold_suffix = f'/{round(1-args.cold_start, 2)}' if args.cold_start > 0 and args.cold_start != 0.2 else ""
    train_mat_path = os.path.join(
        f'{data_root}{other_cold_suffix}', 
        f"{'warm_' if args.cold_start > 0 else ''}train_matrix.npz"
    )
    val_mat_path = os.path.join(f'{data_root}{other_cold_suffix}', f"{'overall_' if args.cold_start > 0 else ''}val_matrix.npz") # TODO: change to cold or overall
    val_user_ids = None
    warm_item_idx, cold_item_idx = None, None
    if args.cold_start > 0:
        info_dict_path = os.path.join(f'{data_root}{other_cold_suffix}', f"info_dict.pkl")
        info_dict = pickle.load(open(info_dict_path, "rb"))
        warm_item_idx = info_dict["warm_item"]
        cold_item_idx = info_dict["cold_item"]
        warm_val_user_ids = info_dict["warm_val_user"]
        cold_val_user_ids = info_dict["cold_val_user"]
        val_user_ids = info_dict["overall_val_user"]
        cold_val_mat_path = os.path.join(f'{data_root}{other_cold_suffix}', f"cold_item_val_matrix.npz")
        cold_val_mat = load_npz(cold_val_mat_path)
        warm_val_mat_path = os.path.join(f'{data_root}{other_cold_suffix}', f"warm_val_matrix.npz")
        warm_val_mat = load_npz(warm_val_mat_path)
    # Get the training data generator
    train_mat = load_npz(train_mat_path)
    

    # Get the validation data generator
    val_mat = load_npz(val_mat_path)

    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Collaborative Data Generator-----\n")

        
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

    # accelerator.wait_for_everyone()
    
    memory = UserItemMemory()
    from data import ITEM_CONTENT_FILES
    tokenizer = DynamicBPETokenizerBatch(vocab_file,
        merges_file,
        num_users,
        num_items,
        memory,
        [],
        [os.path.join(data_root, "item_texts", file) for file in ITEM_CONTENT_FILES]
    )
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")
    train_data_gen = RecommendationGPTTrainGeneratorBatch(tokenizer, train_mat)
    val_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, val_mat,
                                                        test_user_ids=val_user_ids)
    if args.cold_start > 0:
        cold_val_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, cold_val_mat, 
                                                                test_user_ids=cold_val_user_ids)
        warm_val_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, warm_val_mat, 
                                                                test_user_ids=warm_val_user_ids)
    
    '''
        Define the review data generator
    '''
    accelerator.print("-----Begin Obtaining the Review Data Generator-----")
    review_path = os.path.join(data_root, "user_item_texts", f"review{cold_start_suffix}.pkl")
    accelerator.print(f"Loading data from {review_path}...")
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, review_path)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Review Data Generator-----\n")


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
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pretrained GPT Model-----\n")
    

    # /jobs/dfo/jundli/llm4rec/model/pretrained
    '''
        Instantiate the GPT for recommendation content model
    '''
    content_model_root = os.path.join(model_root, "model", dataset, model, "content")
    collab_model_root = os.path.join(model_root, "model", dataset, model, "collab")
    content_weight_root = os.path.join(content_model_root, f"content_model_{load_path_suffix}.pth")
    collab_weight_root = os.path.join(collab_model_root, f"collab_model_{load_path_suffix}.pth")


    accelerator.print("-----Begin Instantiating the Content GPT Model-----")
    content_base_classifier = AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_meta, ignore_mismatched_sizes=True)
    content_base_model = DynamicDPELLM4RecBaseModel(
        config, 
        LLMmodel, 
        memory, 
        AutoTokenizer.from_pretrained(bert_tokenizer_path), # TODO: use local tokenizer
        content_base_classifier, 
        device=device,
        num_item_meta=args.num_meta,
        item_logits_infer=args.item_logits_infer,
        prob_norm=args.prob_norm,
        dataset_name=args.dataset
    )
    
    # content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)
    if fs.exists(content_weight_root):
        # content_model.load_state_dict(torch.load(content_weight_root, map_location=device))
        accelerator.print(f"Loaded content model from {content_weight_root}")
    else:
        accelerator.print(f"Content model not found in {content_weight_root}, initializing from scratch")
        
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Content GPT Model-----\n")
    

    '''
        Instantiate the GPT for recommendation model
    '''
    accelerator.print("-----Begin Instantiating the Rec GPT Model-----")
    
    accelerator.wait_for_everyone()
    collab_base_classifier = AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_meta, ignore_mismatched_sizes=True)
    base_model = MSEDynamicDPELLM4RecBaseModel(
        config, 
        LLMmodel, 
        memory, 
        AutoTokenizer.from_pretrained(bert_tokenizer_path), # TODO: use local tokenizer
        collab_base_classifier, 
        device=device,
        num_item_meta=args.num_meta,
        item_logits_infer=args.item_logits_infer,
        prob_norm=args.prob_norm,
        dataset_name=args.dataset
    )

    rec_model = DynamicCollaborativeGPTwithItemRecommendHead(config, base_model, device=device)
    if fs.exists(collab_weight_root) and not args.skip_pt:
        rec_model.load_state_dict(torch.load(collab_weight_root, map_location=device), strict=False)
        accelerator.print(f"Loaded collab model from {collab_weight_root}")
    else:
        accelerator.print(f"Collab model not found in {collab_weight_root}, initializing from scratch")
        accelerator.print(f"Collab model initialized from scratch")
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Collab GPT Model-----\n")


    '''
        Freeze the parameters of the pretrained GPT2 for content model
    '''
    for name, param in rec_model.named_parameters():
        # we allow only user/item token embeddings to be trained
        if ('user' not in name) and \
           ('item' not in name) and \
           ('classifier' not in name):
            param.requires_grad = False

    accelerator.print("-----Trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))
    
    # accelerator.print("\n-----Non-trainable Parameters-----")
    # for name, param in rec_model.named_parameters():
    #     if not param.requires_grad:
    #         accelerator.print("{} : {}".format(name, param.shape))

    '''
        Set up the training details
    '''
    accelerator.print("-----Begin Setting Up the Training Details-----")
    val_batch_size = 256
    num_epochs = 50


    '''
        Create a data sampler for distributed training
    '''
    accelerator.print("-----Begin Creating the DataLoader-----")

    # Create the training data loader
    train_data_loader = DataLoader(train_data_gen, 
                                   batch_size=batch_size, 
                                   collate_fn=train_data_gen.collate_fn)

    # Create the validation data loader
    # Note that we only do the validation in the main process!
    val_data_loader = DataLoader(val_data_gen, 
                                 batch_size=val_batch_size, 
                                 collate_fn=val_data_gen.collate_fn)
    if args.cold_start > 0:
        cold_val_data_loader = DataLoader(cold_val_data_gen,
                                        batch_size=val_batch_size, 
                                        collate_fn=cold_val_data_gen.collate_fn)
        
        warm_val_data_loader = DataLoader(warm_val_data_gen,
                                        batch_size=val_batch_size, 
                                        collate_fn=warm_val_data_gen.collate_fn)
    
    # Create the review data loader with the custom collate_fn
    review_data_loader = DataLoader(review_data_gen, 
                                    batch_size=batch_size, 
                                    collate_fn=review_data_gen.collate_fn)
    accelerator.print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    rec_model.to(device)
    
    # content_model.train()
    # content_model.to(device)

    # Obtain the optimizer
    optimizer = optim.Adam(rec_model.parameters(), 
                           lr=learning_rate)
    
    # review_optimizer = optim.Adam(content_model.parameters(), 
    #                               lr=learning_rate)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=1)
    # review_scheduler = optim.lr_scheduler.StepLR(review_optimizer, step_size=2, gamma=1)

    # Parallel model, optimizer and data loader with accelerator
    rec_model, optimizer, train_data_loader = accelerator.prepare(
        rec_model, optimizer, train_data_loader
    )
    
    review_best_loss = float('inf')
    best_val_rec_loss = float('inf')
    best_recall_20 = -float('inf')
    best_recall_40 = -float('inf')
    best_NDCG_100 = -float('inf')
    best_sum = -float('inf')
    best_cold_recall_20 = -float('inf')
    best_cold_recall_40 = -float('inf')
    best_cold_NDCG_100 = -float('inf')
    best_cold_sum = -float('inf')
    best_warm_recall_20 = -float('inf')
    best_warm_recall_40 = -float('inf')
    best_warm_NDCG_100 = -float('inf')
    best_warm_sum = -float('inf')

    # The place to save the recommendation model weights
    rec_model_root = os.path.join(model_root, "model", dataset, model, "rec")
    if not fs.exists(rec_model_root):
        fs.makedirs(rec_model_root, exist_ok=True)   
    accelerator.print(f"Weights will be saved to {rec_model_root}!")
    
    # The place to save the content model weights
    accelerator.print(f"Weights will be saved to {content_model_root}!")
    accelerator.print("-----End Setting Up the Training Details-----\n")

    '''
        Define the pretraining loop for the content GPT
    '''
    accelerator.print("-----Begin Rec GPT Pretraining Loop-----")
    for epoch in range(num_epochs):
        # Set the model to the training mode
        rec_model.train()
        train_rec_loss = 0
        regularize_total_loss = 0 
        
        # Initialize tqdm progress bar
        #progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}", ncols=80)
                            #disable=not accelerator.is_local_main_process, ncols=80)
        #for input_ids, target_mat, attention_mask, input_ids_main in progress_bar:
        for input_ids, target_mat, attention_mask, input_ids_main in train_data_loader:
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            target_mat = target_mat.to(device)
            attention_mask = attention_mask.to(device)
            input_ids_main = input_ids_main.to(device)

            accelerator.wait_for_everyone()
            # with torch.no_grad():
            #     content_embeds = torch.cat(
            #         (accelerator.unwrap_model(content_model).base_model.embed(input_ids),
            #          accelerator.unwrap_model(content_model).base_model.embed(input_ids_main)),
            #         axis=1
            #     ).to(device)

            # Forward pass
            outputs = rec_model(input_ids, 
                                target_mat, 
                                attention_mask=attention_mask,
                                regularize=True,
                                lambda_V=lambda_V,
                                main_ids=input_ids_main,
                                content_embeds=None)
            rec_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            accelerator.backward(rec_loss)
            optimizer.step()
            scheduler.step()

            train_rec_loss += rec_loss.item()
            regularize_total_loss += regularize_loss.item()
            #progress_bar.set_postfix({"Rec Loss": rec_loss.item()})

        # Gather the multinomial recommendation loss from different device
        thread_train_rec_loss = torch.tensor([train_rec_loss / len(train_data_loader)]).to(device)
        gathered_train_rec_loss = accelerator.gather(thread_train_rec_loss)
        train_rec_loss = torch.mean(gathered_train_rec_loss)
        accelerator.print(f"Epoch {epoch + 1} - Rec Loss: {train_rec_loss:.4f}")

        # Gather the regularize loss from difference device
        thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(train_data_loader)]).to(device)
        gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
        regularize_average_loss = torch.mean(gathered_regularize_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Set the model to evaluation mode
        rec_model.eval()  
        val_rec_loss = 0
        cur_recall_20 = 0
        cur_recall_40 = 0
        cur_NDCG_100 = 0
        cold_rec_loss = 0
        cur_cold_recall_20 = 0
        cur_cold_recall_40 = 0
        cur_cold_NDCG_100 = 0
        warm_rec_loss = 0
        cur_warm_recall_20 = 0
        cur_warm_recall_40 = 0
        cur_warm_NDCG_100 = 0

        accelerator.wait_for_everyone()
        with torch.no_grad():
            for input_ids, train_mat, target_mat, attention_mask in val_data_loader:
                # Move tensors to the correct device
                input_ids = input_ids.to(device)
                train_mat = train_mat.to(device)
                target_mat = target_mat.to(device)
                attention_mask = attention_mask.to(device)

                # Get item scores and rank them
                rec_loss, item_scores = rec_model(input_ids, 
                                                    target_mat, 
                                                    attention_mask,
                                                    lambda_V=lambda_V)
                
                # Set score of interacted items to the lowest
                item_scores[train_mat > 0] = -float("inf")  

                # Calculate Recall@K and NDCG@K for each user
                target_mat = target_mat.cpu().numpy()
                item_scores = item_scores.cpu().numpy()
                val_rec_loss += rec_loss.item()
                cur_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
                cur_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
                cur_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")
            if args.cold_start > 0:
                for input_ids, train_mat, target_mat, attention_mask in cold_val_data_loader:
                    # Move tensors to the correct device
                    input_ids = input_ids.to(device)
                    train_mat = train_mat.to(device)
                    target_mat = target_mat.to(device)
                    attention_mask = attention_mask.to(device)

                    # Get item scores and rank them
                    rec_loss, item_scores = rec_model(input_ids, 
                                                        target_mat, 
                                                        attention_mask,
                                                        lambda_V=lambda_V)
                    
                    # Set score of interacted items to the lowest
                    item_scores[train_mat > 0] = -float("inf")  

                    # Calculate Recall@K and NDCG@K for each user
                    target_mat = target_mat.cpu().numpy()
                    item_scores = item_scores.cpu().numpy()
                    item_scores[:, warm_item_idx] = -float("inf")
                    cold_rec_loss += rec_loss.item()
                    cur_cold_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
                    cur_cold_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
                    cur_cold_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")

                for input_ids, train_mat, target_mat, attention_mask in warm_val_data_loader:
                    # Move tensors to the correct device
                    input_ids = input_ids.to(device)
                    train_mat = train_mat.to(device)
                    target_mat = target_mat.to(device)
                    attention_mask = attention_mask.to(device)

                    # Get item scores and rank them
                    rec_loss, item_scores = rec_model(input_ids, 
                                                        target_mat, 
                                                        attention_mask,
                                                        lambda_V=lambda_V)
                    
                    # Set score of interacted items to the lowest
                    warm_rec_loss += rec_loss.item()
                    item_scores[train_mat > 0] = -float("inf")  

                    # Calculate Recall@K and NDCG@K for each user
                    target_mat = target_mat.cpu().numpy()
                    item_scores = item_scores.cpu().numpy()
                    item_scores[:, cold_item_idx] = -float("inf")
                    cur_warm_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
                    cur_warm_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
                    cur_warm_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")

        # Calculate average Recall@K and NDCG@K for the validation set
        val_rec_loss /= len(val_data_loader)
        cur_recall_20 /= len(val_data_gen)
        cur_recall_40 /= len(val_data_gen)
        cur_NDCG_100 /= len(val_data_gen)
        cur_sum = cur_recall_20 + cur_recall_40 + cur_NDCG_100
        if args.cold_start > 0:
            cold_rec_loss /= len(cold_val_data_loader)
            cur_cold_recall_20 /= len(cold_val_data_gen)
            cur_cold_recall_40 /= len(cold_val_data_gen)
            cur_cold_NDCG_100 /= len(cold_val_data_gen)
            cur_cold_sum = cur_cold_recall_20 + cur_cold_recall_40 + cur_cold_NDCG_100

            warm_rec_loss /= len(warm_val_data_loader)
            cur_warm_recall_20 /= len(warm_val_data_gen)
            cur_warm_recall_40 /= len(warm_val_data_gen)
            cur_warm_NDCG_100 /= len(warm_val_data_gen)
            cur_warm_sum = cur_warm_recall_20 + cur_warm_recall_40 + cur_warm_NDCG_100
    
        # Update the best metrics
        if val_rec_loss < best_val_rec_loss:
            best_val_rec_loss = val_rec_loss
        if cur_recall_20 > best_recall_20:
            best_recall_20 = cur_recall_20
        if cur_recall_40 > best_recall_40:
            best_recall_40 = cur_recall_40
        if cur_NDCG_100 > best_NDCG_100:
            best_NDCG_100 = cur_NDCG_100
        
        if cur_sum > best_sum:
            best_sum = cur_sum
            if accelerator.is_main_process:
                # Save user embeddings in the main process
                rec_weight_path = os.path.join(rec_model_root, f"rec_model_{path_suffix}.pth")
                torch.save(accelerator.unwrap_model(rec_model).state_dict(), rec_weight_path)
                accelerator.print(f"Best rec model saved to {rec_weight_path}")
        accelerator.wait_for_everyone()

        accelerator.print(f"Best model saved to {rec_model_root}")
        accelerator.print(f"Train Rec Loss: {train_rec_loss:.4f}")
        accelerator.print(f"Val Rec Loss: {val_rec_loss:.4f} / Best Val Rec Loss: {best_val_rec_loss:.4f}")
        accelerator.print(f"Cur Recall@20: {cur_recall_20:.4f} / Best Recall@20: {best_recall_20:.4f}")
        accelerator.print(f"Cur Recall@40: {cur_recall_40:.4f} / Best Recall@40: {best_recall_40:.4f}")
        accelerator.print(f"Cur NDCG@100: {cur_NDCG_100:.4f} / Best NDCG@100: {best_NDCG_100:.4f}")  
        if args.cold_start > 0:
            accelerator.print(f"Cold Rec Loss: {cold_rec_loss:.4f}")
            accelerator.print(f"Cur Cold&Warm Recall@20: {cur_cold_recall_20:.4f}&{cur_warm_recall_20:.4f} / Best Cold&Warm Recall@20: {best_cold_recall_20:.4f}&{best_warm_recall_20:.4f}")
            accelerator.print(f"Cur Cold&Warm Recall@40: {cur_cold_recall_40:.4f}&{cur_warm_recall_40:.4f} / Best Cold&Warm Recall@40: {best_cold_recall_40:.4f}&{best_warm_recall_40:.4f}")
            accelerator.print(f"Cur Cold&Warm NDCG@100: {cur_cold_NDCG_100:.4f}&{cur_warm_NDCG_100:.4f} / Best Cold&Warm NDCG@100: {best_cold_NDCG_100:.4f}&{best_warm_NDCG_100:.4f}")
            accelerator.print(f"Cur Cold&Warm Sum: {cur_cold_sum:.4f}&{cur_warm_sum:.4f} / Best Cold&Warm Sum: {best_cold_sum:.4f}&{best_warm_sum:.4f}")
    
        review_total_loss = 0
        regularize_total_loss = 0
        

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
            content_weight_path = os.path.join(content_model_root, f"content_model_{path_suffix}.pth")
            # if accelerator.is_main_process:
                # torch.save(accelerator.unwrap_model(content_model).state_dict(), content_weight_path)
            accelerator.print(f"Best content model saved to {content_model_root}")
        accelerator.wait_for_everyone()
            
            
        
    accelerator.print("-----End Rec GPT Pretraining Loop-----")


if __name__ == "__main__":
    main()
import re
import os
import sys
import json
import pickle
import fsspec
import random
import argparse
from tqdm import tqdm
import scipy.sparse as sp

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config, AutoModel
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
import copy
from model import (
    UserItemMemory,
    DynamicDPELLM4RecBaseModel,
    DynamicCollaborativeGPTwithItemRecommendHead,
    MSEDynamicDPELLM4RecBaseModel
)
from tokenizer import DynamicBPETokenizerBatch

from data import (
    RecommendationGPTTestGeneratorBatch,
)

from warm_ft import Recall_at_k, NDCG_at_k 
    
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
    # Parse the command line arguments
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--model", type=str,
        help="specify the size of the model")
    parser.add_argument("--num_meta", type=int,
        help="number of meta_embeddings")
    parser.add_argument('--shared_root', type=str, default="",)
    parser.add_argument('--model_root', type=str, default="",)
    parser.add_argument('--local_root', type=str, default="tmp",)
    parser.add_argument('-c', '--cold_start', type=float, default=0.0,)
    parser.add_argument('--item_logits_infer', type=str, default="classifier",)
    parser.add_argument('-p', '--prob_norm', type=str, default="softmax",)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,)
    parser.add_argument('-cp', '--cold_start_type', type=str, default="random",)
    parser.add_argument('--calibration', type=bool, default=False, action='store_true',)
    args = parser.parse_args()
    path_suffix = f"{args.lambda_V}_{args.num_meta}_{args.cold_start}_{args.item_logits_infer}_{args.learning_rate}"
    
    print(f'path suffix: {path_suffix}')
    shared_root = args.shared_root
    model_root = args.model_root
    local_root = args.local_root
    dataset = args.dataset
    num_meta = args.num_meta
    model = args.model
    prob_norm = args.prob_norm
    model_path = f'/shared/public/models/'
    rec_model_root = os.path.join(model_root, "model", dataset, model, "rec")
    # cold_start_suffix = f"_{args.cold_start}" if args.cold_start > 0 else ""
    cold_start_suffix = ""
    cold_flag = args.cold_start > 0
    if not os.path.exists(local_root):
        os.makedirs(local_root, exist_ok=True)
    
    
    assert dataset in ["beauty", "toys", "sports", "yelp", "company"]
    assert model in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    lambda_V = float(args.lambda_V)

    host = "hdfs://ltx1-holdem//"
    port = 8443
    fs = fsspec.filesystem('file')
    
    # Define the device
    device = "cuda"

    # Define the number of GPUs to be used
    num_gpus = torch.cuda.device_count()
    print(f"num_gpus: {num_gpus}")
    
    '''
        Get the basic information of the dataset
    '''
    print("-----Begin Obtaining Dataset Info-----")
    data_root = os.path.join(shared_root, "dataset", dataset)
    # meta_path = os.path.join(data_root, f"meta{cold_start_suffix}.pkl")
    print("-----Begin Obtaining the Collaborative Data Generator-----")
    
    
    other_cold_suffix = f'/{round(1-args.cold_start, 2)}' if args.cold_start > 0 and args.cold_start != 0.2 else ""
    train_mat_path = os.path.join(
        f'{data_root}{other_cold_suffix}', 
        f"{'warm_' if args.cold_start > 0 else ''}train_matrix.npz"
    )
    print(f"Loading data from {train_mat_path}...")
    
    info_dict_path = os.path.join(f'{data_root}{other_cold_suffix}', f"info_dict.pkl")
    test_mat_path = os.path.join(f'{data_root}{other_cold_suffix}', f"overall_test_matrix.npz" if cold_flag else f"test_matrix.npz")
    warm_test_mat_path = os.path.join(f'{data_root}{other_cold_suffix}', f"warm_test_matrix.npz")
    if dataset == 'yelp' or args.cold_start > 0.2:
        test_suffix = "cold_item_test_matrix.npz"
    else:
        test_suffix = "cold_test_matrix.npz"
    cold_test_mat_path = os.path.join(f'{data_root}{other_cold_suffix}', test_suffix)
    print(f"Loading data from {test_mat_path}...")
    
    train_mat = load_npz(train_mat_path)
    test_mat = load_npz(test_mat_path)
    warm_test_mat = load_npz(warm_test_mat_path)
    cold_test_mat = load_npz(cold_test_mat_path)
    info_dict = pickle.load(open(info_dict_path, "rb"))
    warm_user_ids = info_dict["warm_test_user"]
    cold_user_ids = info_dict["cold_test_user"]
    user_ids = info_dict["overall_test_user"] if cold_flag else None
    warm_item_idx = info_dict["warm_item"]
    cold_item_idx = info_dict["cold_item"]
    
    print(train_mat.shape, test_mat.shape, warm_test_mat.shape, cold_test_mat.shape)
    
    print("Success!")
    print("-----End Obtaining the Collaborative Data Generator-----\n")

    # with fsspec.open(meta_path, "rb") as f:
    #     meta_data = pickle.load(f)
        
    num_users = train_mat.shape[0]
    num_items = train_mat.shape[1]
    print(f"num_users: {num_users}")
    print(f"num_items: {num_items}")
    print("-----End Obtaining Dataset Info-----\n")


    '''
        Obtain the tokenizer with user/item tokens
    '''
    vocab_file = os.path.join(model_path, model, "vocab.json")
    merges_file = os.path.join(model_path, model, "merges.txt")
    bert_model_path = os.path.join(model_path, "bert-base-uncased")
    bert_tokenizer_path = os.path.join(model_path, "bert-base-uncased")
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
    print("Success!")
    print("-----End Obtaining the Tokenizer-----\n")

    if cold_flag:
        warm_test_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, warm_test_mat, test_user_ids=warm_user_ids)
        cold_test_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, cold_test_mat, test_user_ids=cold_user_ids)
    else:
        test_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, test_mat, test_user_ids=user_ids)
    '''
        Obtain the testing data generator
    '''
    


    '''
        Extend the config of the original GPT model
    '''
    print("-----Begin Setting Up the Config-----")
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
    print("Success!")
    print("-----End Setting Up the Config-----\n")


    '''
        Instantiate the pretrained GPT2 model
    '''
    print("-----Begin Instantiating the Pretrained GPT Model-----")
    LLMmodel = AutoModel.from_pretrained(os.path.join(model_path, model))
    print("Success!")
    print("-----End Instantiating the Pretrained GPT Model-----\n")
    # print(os.listdir(rec_model_root))
    rec_weight_path = os.path.join(rec_model_root, f"rec_model_{path_suffix}.pth")

    '''
        Instantiate the GPT for recommendation content model
    '''
    print("-----Begin Instantiating the Content GPT Model-----")
    pretrained_root = os.path.join(model_root, "model", dataset, model, "rec")
    base_classifier = AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_meta, ignore_mismatched_sizes=True)
    
    base_model = MSEDynamicDPELLM4RecBaseModel(
        config, 
        LLMmodel, 
        memory, 
        AutoTokenizer.from_pretrained(bert_tokenizer_path), # TODO: use local tokenizer
        base_classifier, 
        device=device,
        num_item_meta=num_meta,
        item_logits_infer=args.item_logits_infer,
        prob_norm=prob_norm,
        dataset_name=dataset
    )



    rec_model = DynamicCollaborativeGPTwithItemRecommendHead(config, base_model, device=device)
    print(f'load model from {rec_weight_path}')
    rec_model.load_state_dict(torch.load(rec_weight_path, map_location=device), strict=False)
    print("-----End Instantiating the Content GPT Model-----\n")

    
    '''
        Create a data sampler for distributed training
    '''
    print("-----Begin Creating the DataLoader-----")

    # Create the testing data loader
    # Note that we only do the testing in the main process!
    batch_size = 256
    if cold_flag:
        warm_test_data_loader = DataLoader(warm_test_data_gen,
                                        batch_size=batch_size,
                                        collate_fn=warm_test_data_gen.collate_fn)
        cold_test_data_loader = DataLoader(cold_test_data_gen,
                                            batch_size=batch_size,
                                            collate_fn=cold_test_data_gen.collate_fn)
    else:
        test_data_loader = DataLoader(test_data_gen, 
                                  batch_size=batch_size, 
                                  collate_fn=test_data_gen.collate_fn)
    print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    rec_model.to(device)
    
    # Set the model to evaluation mode
    rec_model.eval()  
    cur_recall_20 = 0
    cur_recall_40 = 0
    cur_NDCG_100 = 0
    warm_cur_recall_20 = 0
    warm_cur_recall_40 = 0
    warm_cur_NDCG_100 = 0
    cold_cur_recall_20 = 0
    cold_cur_recall_40 = 0
    cold_cur_NDCG_100 = 0
    with torch.no_grad():
        
        print(f"Final Testing Results:")
        print(f"Recall@20: {cur_recall_20:.4f}")
        print(f"Recall@40: {cur_recall_40:.4f}")
        print(f"NDCG@100: {cur_NDCG_100:.4f}")
        if cold_flag:
            for warm_input_ids, warm_train_mat, warm_target_mat, warm_attention_mask in warm_test_data_loader:
                # Move tensors to the correct device
                warm_input_ids = warm_input_ids.to(device)
                warm_train_mat = warm_train_mat.to(device)
                warm_target_mat = warm_target_mat.to(device)
                warm_attention_mask = warm_attention_mask.to(device)

                # Get item scores and rank them
                warm_rec_loss, warm_item_scores = rec_model(warm_input_ids, 
                                                    warm_target_mat, 
                                                    warm_attention_mask,
                                                    lambda_V=lambda_V)

                # Calculate Recall@K and NDCG@K for each user
                warm_target_mat = warm_target_mat.cpu().numpy()
                warm_item_scores = warm_item_scores.cpu().numpy()
                warm_item_scores[:, cold_item_idx] = -float("inf")
                warm_cur_recall_20 += Recall_at_k(warm_target_mat, warm_item_scores, k=20, agg="sum")
                warm_cur_recall_40 += Recall_at_k(warm_target_mat, warm_item_scores, k=40, agg="sum")
                warm_cur_NDCG_100 += NDCG_at_k(warm_target_mat, warm_item_scores, k=100, agg="sum")
            
            warm_cur_recall_20 /= len(warm_test_data_gen)
            warm_cur_recall_40 /= len(warm_test_data_gen)
            warm_cur_NDCG_100 /= len(warm_test_data_gen)
            
            print(f"Warm Testing Results:")
            print(f"Recall@20: {warm_cur_recall_20:.4f}")
            print(f"Recall@40: {warm_cur_recall_40:.4f}")
            print(f"NDCG@100: {warm_cur_NDCG_100:.4f}")

            for cold_input_ids, cold_train_mat, cold_target_mat, cold_attention_mask in cold_test_data_loader:
                # Move tensors to the correct device
                cold_input_ids = cold_input_ids.to(device)
                cold_train_mat = cold_train_mat.to(device)
                cold_target_mat = cold_target_mat.to(device)
                cold_attention_mask = cold_attention_mask.to(device)

                # Get item scores and rank them
                cold_rec_loss, cold_item_scores = rec_model(cold_input_ids, 
                                                    cold_target_mat, 
                                                    cold_attention_mask,
                                                    lambda_V=lambda_V)
                
                cold_item_scores[cold_train_mat > 0] = -float("inf")
                # Calculate Recall@K and NDCG@K for each user
                cold_target_mat = cold_target_mat.cpu().numpy()
                cold_item_scores = cold_item_scores.cpu().numpy()
                cold_item_scores[:, warm_item_idx] = -float("inf")

                cold_cur_recall_20 += Recall_at_k(cold_target_mat, cold_item_scores, k=20, agg="sum")
                cold_cur_recall_40 += Recall_at_k(cold_target_mat, cold_item_scores, k=40, agg="sum")
                cold_cur_NDCG_100 += NDCG_at_k(cold_target_mat, cold_item_scores, k=100, agg="sum")

            cold_cur_recall_20 /= len(cold_test_data_gen)
            cold_cur_recall_40 /= len(cold_test_data_gen)
            cold_cur_NDCG_100 /= len(cold_test_data_gen)
            
            print(f"Cold Testing Results:")
            print(f"Recall@20: {cold_cur_recall_20:.4f}")
            print(f"Recall@40: {cold_cur_recall_40:.4f}")
            print(f"NDCG@100: {cold_cur_NDCG_100:.4f}")
        
        else:     
        # Non-Cold Start Results
            for input_ids, train_mat, target_mat, attention_mask in test_data_loader:
                input_ids = input_ids.to(device)
                train_mat = train_mat.to(device)
                target_mat = target_mat.to(device)
                attention_mask = attention_mask.to(device)

                rec_loss, item_scores = rec_model(
                    input_ids, 
                    target_mat, 
                    attention_mask,
                    lambda_V=lambda_V
                )

                item_scores[train_mat > 0] = -float("inf")  
                # Calculate Recall@K and NDCG@K for each user
                target_mat = target_mat.cpu().numpy()
                item_scores = item_scores.cpu().numpy()

                cur_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
                cur_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
                cur_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")

            cur_recall_20 /= len(test_data_gen)
            cur_recall_40 /= len(test_data_gen)
            cur_NDCG_100 /= len(test_data_gen)
            print(f"Testing Results:")
            print(f"Recall@20: {cur_recall_20:.4f}")
            print(f"Recall@40: {cur_recall_40:.4f}")
            print(f"NDCG@100: {cur_NDCG_100:.4f}")

if __name__ == "__main__":
    main()

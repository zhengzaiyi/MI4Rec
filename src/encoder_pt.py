from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import random
import argparse
import pickle as pkl
from tqdm import tqdm
import os
import numpy as np
from accelerate import Accelerator
from functools import partial
from transformers import AutoTokenizer

def generate_text_input(segments):
    if len(segments) == 0:
        return ""
    return 'User Reviews: ' + ''.join(f'{i + 1}: {r}\n' for i, r in enumerate(segments))  

def infonce_loss(anchor, positive, negative, temperature=0.07):
    """
    InfoNCE loss implementation for contrastive learning
    Args:
        anchor: anchor embeddings [batch_size, embedding_dim]
        positive: positive embeddings [batch_size, embedding_dim]  
        negative: negative embeddings [batch_size, embedding_dim]
        temperature: temperature parameter for scaling
    """
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)
    
    # Compute cosine similarities
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature  # [batch_size]
    neg_sim = torch.sum(anchor * negative, dim=1) / temperature  # [batch_size]
    
    # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim)))
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)  # [batch_size, 2]
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)  # positive is always label 0
    
    return F.cross_entropy(logits, labels)

def generate_triplet_data(items_texts, dropout_prob=0.3):
    examples = []
    count = 0
    for i, segments in enumerate(items_texts):
        if len(segments) == 0:
            count += 1
            continue
        anchor = generate_text_input(segments)

        positive_segments = [s for s in segments if random.random() > dropout_prob]
        if len(positive_segments) == 0:
            positive_segments = [random.choice(segments)]
        positive = generate_text_input(positive_segments)

        negative_idx = random.choice([j for j in range(len(items_texts)) if j != i])
        negative = generate_text_input(items_texts[negative_idx])

        examples.append(InputExample(texts=[anchor, positive, negative]))
    print('zero length segments:', count)
    return examples

def collate_fn(batch, tokenizer, max_length=128):
    anchor_texts = [ex.texts[0] for ex in batch]
    positive_texts = [ex.texts[1] for ex in batch]
    negative_texts = [ex.texts[2] for ex in batch]
    return {
        "anchor": tokenizer(anchor_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
        "positive": tokenizer(positive_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
        "negative": tokenizer(negative_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="beauty")
    parser.add_argument("--data_root", type=str, default="/shared/user/llm4rec/dataset")
    parser.add_argument("--file_name", type=str, default="item2review")
    parser.add_argument("--output_path", type=str, default="/shared/user/llm4rec/models")
    parser.add_argument("--emb_path", type=str, default='/shared/user/embs')
    parser.add_argument("--pretrain", action='store_true')
    args = parser.parse_args()

    data_root = f"{args.data_root}/{args.dataset}"
    with open(f"{data_root}/{args.file_name}.pkl", "rb") as f:
        item2review = pkl.load(f)
    random.seed(42)

    accelerator = Accelerator()
    device = accelerator.device

    train_examples = generate_triplet_data(item2review, dropout_prob=args.dropout_prob)
    model_name = "/shared/public/models/stella_en_400M_v5"
    model = SentenceTransformer(
        model_name, trust_remote_code=True,
        config_kwargs={
            "use_memory_efficient_attention": False,
            "unpad_inputs": False
        }
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    train_dataloader = DataLoader(
        train_examples, shuffle=True, 
        batch_size=args.batch_size, drop_last=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_dataloader:
            def to_device(d): return {k: v.to(device) for k, v in d.items()}

            anchor = model(to_device(batch["anchor"]))["sentence_embedding"]
            positive = model(to_device(batch["positive"]))["sentence_embedding"]
            negative = model(to_device(batch["negative"]))["sentence_embedding"]

            loss = infonce_loss(anchor, positive, negative)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
            model.save(f"{args.output_path}/stella_pretrained_{args.dataset}_dropout{args.dropout_prob}")
        accelerator.wait_for_everyone()

    # Generate Embeddings
    if accelerator.is_main_process:
        model.eval()
        num_items = len(item2review)
        list_item2review = [''] * num_items
        for item_id, review in item2review.items():
            list_item2review[item_id] = generate_text_input(review)

        item_review_embeddings = []
        print("Generating item review embeddings...")
        for i in range(0, len(list_item2review) // args.batch_size + 1):
            batch_slice = slice(i * args.batch_size, min((i + 1) * args.batch_size, len(list_item2review)))
            batch = list_item2review[batch_slice]
            tmp_review_embeddings = model.encode(batch, show_progress_bar=False)
            item_review_embeddings.append(tmp_review_embeddings)
        print("Finished generating item review embeddings.")
        item_review_embeddings = np.concatenate(item_review_embeddings, axis=0)
        with open(os.path.join(args.emb_path, f"{args.dataset}_item_review_embeddings.pkl"), "wb") as file:
            pkl.dump(item_review_embeddings, file)
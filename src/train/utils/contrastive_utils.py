import torch
from tqdm.autonotebook import tqdm


def contrastive_loss(features_1, features_2, temperature=0.05):
    device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
    batch_size = features_1.shape[0]
    features = torch.cat([features_1, features_2], dim=0)
    # Mask for selecting only the negative pairs
    mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    mask = mask.repeat(2, 2)
    mask = ~mask
    
    # positive pairs
    pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    # all-to-all cosine similarity matrix
    all_sim = torch.mm(features, features.t().contiguous())
    neg = torch.exp(all_sim / temperature).masked_select(mask).view(2*batch_size, -1) # (2B, 2B-2)
    
    # importance of the negative samples
    negimp = neg / neg.mean(dim=-1, keepdim=True)
    loss_pos = - torch.log(pos / ((negimp * neg).sum(1) + pos)).sum() / (2*batch_size)
    
    return loss_pos


def tokenize_sequences(batch, max_length, tokenizer):
    feat = tokenizer.batch_encode_plus(
        batch, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return feat
        
        
def compute_embeddings(model, sequences, tokenizer, batch_size=32, max_length=256):
    all_embeddings = []
    pbar = tqdm(total=len(sequences), leave=False)
    for i in range(0, len(sequences), batch_size):
        pbar.update(batch_size)
        batch = sequences[i:i+batch_size]
        tokenized_batch = tokenize_sequences(batch, max_length, tokenizer)
        input_ids = tokenized_batch["input_ids"].to("cuda")
        attention_mask = tokenized_batch["attention_mask"].to("cuda")
        with torch.no_grad():
            embeddings = model.get_mean_embeddings(input_ids, attention_mask)
        all_embeddings.append(embeddings.cpu())
    pbar.close()
    
    return torch.cat(all_embeddings, dim=0)
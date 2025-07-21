import torch
from torch.utils.data import Dataset
import csv


class PairDataset(Dataset):
    def __init__(self, fname: str):
        delimiter = ","
        with open(fname) as csvfile:
            data = list(csv.reader(csvfile, delimiter=delimiter))
        self.seq1 = [d[0] for d in data] 
        self.seq2 = [d[1] for d in data]
        assert len(self.seq1) == len(self.seq2)
        
    def __len__(self):
        return len(self.seq1)

    def __getitem__(self, idx):
        return {'seq1': self.seq1[idx], 'seq2': self.seq2[idx]}


def get_batch_token(seq, max_length, tokenizer, insert_whitespace=False):
    
    def add_space(x):
        return " ".join(x)
    
    # Only for prot_bert
    if insert_whitespace:
        seq = list(map(add_space, seq))
    
    token_feat = tokenizer.batch_encode_plus(
        seq, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    
    return token_feat


class PairwiseInputCollator:
    def __init__(self, tokenizer, max_length=None, insert_whitespace=False):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.insert_whitespace = insert_whitespace
        
    def __call__(self, batch):
        text1 = [item['seq1'] for item in batch]
        text2 = [item['seq2'] for item in batch]
        feat1 = get_batch_token(text1, self.max_length, self.tokenizer, insert_whitespace=self.insert_whitespace)
        feat2 = get_batch_token(text2, self.max_length, self.tokenizer, insert_whitespace=self.insert_whitespace)

        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return {
            "input_ids": input_ids.cuda(),
            "attention_mask": attention_mask.cuda()
        }
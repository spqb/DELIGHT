import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import PeftModel

IMPLEMENTED_MODELS = ["Rostlab/prot_bert", "facebook/esm2_t30_150M_UR50D", "facebook/esm2_t12_35M_UR50D", "facebook/esm2_t6_8M_UR50D"]


class ContrastiveLM(nn.Module):
    def __init__(self, feat_dim: int=128, backbone: str = "RostLab/prot_bert"):
        super(ContrastiveLM, self).__init__()
        
        if backbone not in IMPLEMENTED_MODELS:
            raise ValueError(f"Model {backbone} not implemented. Choose from {IMPLEMENTED_MODELS}")
        self.backbone = AutoModel.from_pretrained(backbone, trust_remote_code=True)
        self.emb_size = self.backbone.config.hidden_size
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
    
    def device(self):
        return next(self.parameters()).device
        
    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        
        return feat1, feat2
    
    
    def get_mean_embeddings(self, input_ids, attention_mask):
        model_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(model_output[0] * attention_mask, 1) / torch.sum(attention_mask, 1)
        
        return embeddings
    
    def get_mean_embeddings_with_heads(self, input_ids, attention_mask):
        model_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(model_output[0] * attention_mask, 1) / torch.sum(attention_mask, 1)
        embeddings = F.normalize(self.contrast_head(embeddings), dim=1)
        
        return embeddings
        
        
    def forward(self, input_ids, attention_mask, task_type="train"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)
            model_output_1 = self.backbone(input_ids=input_ids_1, attention_mask=attention_mask_1)
            model_output_2 = self.backbone(input_ids=input_ids_2, attention_mask=attention_mask_2)
            attention_mask_1 = attention_mask_1.unsqueeze(-1)
            attention_mask_2 = attention_mask_2.unsqueeze(-1)
            # Take the mean of the last hidden state
            mean_output_1 = torch.sum(model_output_1[0] * attention_mask_1, 1) / torch.sum(attention_mask_1, 1)
            mean_output_2 = torch.sum(model_output_2[0] * attention_mask_2, 1) / torch.sum(attention_mask_2, 1)
            feat1, feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            
            return feat1, feat2, mean_output_1, mean_output_2
        
        
    def save(self, output_dir: str):
        self.backbone.save_pretrained(output_dir)
        torch.save(self.contrast_head, os.path.join(output_dir, "contrast_head.pt"))
        
        
    def load_from_Peft(self, path: str):
        self.backbone = PeftModel.from_pretrained(self.backbone, path)
        self.backbone = self.backbone.merge_and_unload()
        file_head = os.path.join(path, "contrast_head.pt")
        if os.path.exists(file_head):
            self.contrast_head = torch.load(file_head)
        
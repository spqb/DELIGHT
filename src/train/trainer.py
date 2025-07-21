from transformers import Trainer
from train.utils.contrastive_utils import contrastive_loss
from torch.utils.data import DataLoader, WeightedRandomSampler


class ContrastiveTrainer(Trainer):
    def __init__(self, sequence_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sequence_weights is not None:
            assert len(sequence_weights) == len(self.train_dataset), "Sequence weights must be the same length as the dataset"
        self.sequence_weights = sequence_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        feat_1, feat2, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = contrastive_loss(feat_1, feat2)
        return (loss, (feat_1, feat2)) if return_outputs else loss
    
    def get_train_dataloader(self):
        if self.sequence_weights is None:
            return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, collate_fn=self.data_collator, shuffle=True)
        else:
            sampler = WeightedRandomSampler(self.sequence_weights, len(self.sequence_weights), replacement=False)
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, collate_fn=self.data_collator, sampler=sampler)
            return dataloader
        
    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        dataloader = DataLoader(self.eval_dataset, batch_size=self.args.train_batch_size, collate_fn=self.data_collator, shuffle=False)
        return dataloader
    
    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self.model.save(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
            
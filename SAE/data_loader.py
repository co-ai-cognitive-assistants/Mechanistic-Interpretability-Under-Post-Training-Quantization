import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Iterator, Optional
from config import DatasetConfig

class DataLoader:
    def __init__(self, config: DatasetConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Load dataset
        load_kwargs = {
            "split": "train",
            "streaming": True,
            "trust_remote_code": True
        }
        
        if self.config.subset:
            self.dataset = load_dataset(self.config.path, self.config.subset, **load_kwargs)
        else:
            self.dataset = load_dataset(self.config.name, **load_kwargs)
            
        # Shuffle with seed
        self.dataset = self.dataset.shuffle(seed=self.config.seed)
        
        self.iterator = iter(self.dataset)
        self.total_tokens_yielded = 0

    def __iter__(self) -> Iterator[torch.Tensor]:
        buffer = []
        
        while self.total_tokens_yielded < self.config.num_tokens:
            try:
                item = next(self.iterator)
            except StopIteration:
                # Restart iterator if dataset runs out but we need more tokens
                self.iterator = iter(self.dataset)
                item = next(self.iterator)

            text = item['text']
            tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)['input_ids'][0]
            
            # Filter out very short sequences
            if len(tokens) < 10:
                continue
                
            buffer.append(tokens)
            
            if len(buffer) >= self.config.llm_batch_size:
                # Let's ensure uniform length for the batch
                max_len = max(len(t) for t in buffer)
                padded_batch = []
                for t in buffer:
                    pad_len = max_len - len(t)
                    if pad_len > 0:
                        # Pad with eos_token_id or 0
                        pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
                        padded_batch.append(torch.cat([t, torch.full((pad_len,), pad_id, dtype=torch.long)]))
                    else:
                        padded_batch.append(t)
                
                batch_tensor = torch.stack(padded_batch)
                yield batch_tensor
                
                self.total_tokens_yielded += batch_tensor.numel()
                buffer = []

    def get_token_count(self):
        return self.total_tokens_yielded
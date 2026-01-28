import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Union
import os

@dataclass
class DatasetConfig:
    name: str
    path: str
    num_tokens: int
    batch_size: int
    llm_batch_size: int
    seed: int
    subset: Optional[str] = None

@dataclass
class SAEConfig:
    architecture: str
    activation_dim: int
    total_dict_size: int
    nested_groups: int
    group_fractions: List[float]
    activation_fn: str
    sparsity_k: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    optimizer: str
    reconstruction_weight: float = 1.0

@dataclass
class WandbConfig:
    project: str
    entity: str
    tags: List[str]
    log_interval: int
    run_label: Optional[str] = None

@dataclass
class Config:
    models: List[str]
    methods: List[str]
    layer_index: int
    activation_site: str
    dataset: DatasetConfig
    sae: SAEConfig
    wandb: WandbConfig

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        # Calculate group sizes from fractions if needed, or validate them
        # For now just load as is.
        
        return cls(
            models=cfg_dict['models'],
            methods=cfg_dict['methods'],
            layer_index=cfg_dict['layer_index'],
            activation_site=cfg_dict['activation_site'],
            dataset=DatasetConfig(**cfg_dict['dataset']),
            sae=SAEConfig(**cfg_dict['sae']),
            wandb=WandbConfig(**cfg_dict['wandb'])
        )

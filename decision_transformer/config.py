from dataclasses import dataclass
import torch
from typing import Tuple


@dataclass
class dt_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    state_dim: int = 17
    action_dim: int = 6

    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    sequence_length: int = 20
    episode_length: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    learning_rate: float = 8e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad_norm: float = 0.25
    batch_size: int = 4096
    training_steps: int = 1_000_000
    warmup_steps: int = 10_000
    reward_scale: float = 1e-3
    num_workers: int = 4 # if doesn't work, switch to 0

    project: str = "DecisionTransformer"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
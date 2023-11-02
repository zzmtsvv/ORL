import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class atac_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42
    max_timesteps: int = int(3e5)
    
    max_action : float = 1.0

    state_dim: int = 17
    action_dim: int = 6

    buffer_size: int = 1_000_000
    fast_lr: float = 5e-4
    slow_lr: float = 5e-7
    hidden_dim: int = 256
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005

    critic_ln: bool = True

    beta: float = 64
    omega: float = 0.5
    weight_norm_constraint: float = 100

    project: str = "ATAC"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
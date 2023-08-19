import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class rebrac_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42
    max_timesteps: int = int(3e5)
    
    max_action : float = 1.0

    buffer_size: int = 1_000_000
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    hidden_dim: int = 256
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    actor_bc_coef: float = 0.4
    critic_bc_coef: float = 0.0
    critic_ln: bool = True
    normalize: bool = False

    project: str = "ReBRAC"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)

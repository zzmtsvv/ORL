import torch
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class calql_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    env: str = "halfcheetah-medium-expert-v0"
    seed: int = 42
    eval_seed: int = 42
    eval_frequency: int = 5000
    num_episodes: int = 10
    offline_iterations: int = 1000000
    online_iterations: int = 1000000
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    state_dim: int = 0  # fill to match the desired environment attributes
    action_dim: int = 0
    hidden_dim: int = 256
    max_action: float = 1.0
    target_entropy: int  # = -np.prod(action_shape).item()

    buffer_size: int = 2000000
    batch_size: int = 256
    discount_factor: float = 0.99
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True
    backup_entropy: bool = False
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 5e-3  # Target network update rate

    target_update_period: int = 1
    bc_steps: int = 100000  # Number of BC steps at start
    alpha: float = 5.0
    alpha_online: float = 10.0
    num_actions: int = 10
    use_max_target_backup: bool = False  # Use max target backup
    orthogonal_init: bool = True


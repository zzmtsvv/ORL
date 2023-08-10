import torch
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class calql_config:
    # Experiment
    project: str = "cal_ql"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    env: str = "halfcheetah-medium-v2"
    group: str = env
    seed: int = 42
    eval_seed: int = 42
    eval_frequency: int = 5000
    num_episodes: int = 10
    max_episode_steps = 1000
    offline_iterations: int = 1000000
    online_iterations: int = 1000000
    checkpoints_path: Optional[str] = "cal_ql.pt"  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    state_dim: int = 17
    action_dim: int = 6
    
    hidden_dim: int = 256
    max_action: float = 1.0
    target_entropy: int = -action_dim

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
    importance_sampling: bool = True
    use_lagrange: bool = False
    target_action_gap: float = -1.0  # Action gap
    temperature: float = 1.0  # temperature
    use_max_target_backup: bool = False  # Use max target backup
    clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True
    normalize_states: bool = True
    reward_scale: float = 1.0  #  for normalization
    reward_bias: float = 0.0  # for normalization

    mixing_ratio: float = 0.5  # Data mixing ratio for online tuning
    is_sparse_reward: bool = False  # Use sparse reward


if __name__ == "__main__":
    print(np.log(0.5))

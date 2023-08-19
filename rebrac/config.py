import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class rebrac_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(1e3)  # How often (time steps) we evaluate
    num_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(3e5)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    
    max_action : float = 1.0

    # TD3
    buffer_size: int = 1_000_000  # Replay buffer size
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    hidden_dim: int = 256
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates

    # ReBRAC
    actor_bc_coef: float = 0.4
    critic_bc_coef: float = 0.0
    critic_ln: bool = True
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward

    # Wandb logging
    project: str = "ReBRAC"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)

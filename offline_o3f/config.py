from dataclasses import dataclass
import torch


@dataclass
class o3f_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42
    max_timesteps: int = int(1e6)
    
    max_action : float = 1.0

    action_dim: int = 6
    state_dim: int = 17

    buffer_size: int = 1_000_000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    hidden_dim: int = 256
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005

    critic_ln: bool = True
    num_critics: int = 5
    normalize: bool = True
    standard_deviation: float = 0.2
    num_action_candidates: int = 100

    project: str = "offline_O3F"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
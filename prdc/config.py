from dataclasses import dataclass
import torch


@dataclass
class prdc_config:
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
    hidden_dim: int = 256
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    alpha: float = 40.0
    beta: float = 2
    k: int = 1
    critic_ln: bool = True
    normalize: bool = True

    project: str = "PRDC"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)

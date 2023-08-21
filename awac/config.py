from dataclasses import dataclass
import torch


@dataclass
class awac_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    awac_lambda: float = 0.3333
    gamma: float = 0.99
    hidden_dim: int = 256
    max_action: float = 1.0
    max_timesteps: int = int(3e5)
    tau: float = 5e-3
    exp_adv_max: float = 100.0

    project: str = "AWAC"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
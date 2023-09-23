import torch
from dataclasses import dataclass


@dataclass
class dql_config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    state_dim: int = 17
    action_dim: int = 6

    actor_update_freq: int = 5
    steps_not_updating_actor_target: int = 1000

    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    discount: float = 0.99
    hidden_dim: int = 256
    max_action: float = 1.0
    max_timesteps: int = 1_000_000
    tau: float = 5e-3

    T: int = 5
    eta: float = 1.0
    grad_norm: float = 9.0  # 1.0

    project: str = "DiffusionQL"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
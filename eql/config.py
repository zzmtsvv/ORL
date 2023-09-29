from dataclasses import dataclass
import torch


@dataclass
class eql_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    state_dim: int = 17
    action_dim: int = 6

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_func_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    discount: float = 0.99
    hidden_dim: int = 256
    max_action: float = 1.0
    max_timesteps: int = int(3e5)
    iql_tau: float = 0.7
    tau: float = 5e-3
    beta: float = 3.0
    adv_max: float = 100.0

    layer_norm: bool = True
    alpha: float = 2.0
    beta: float = 10.0

    project: str = "EQL"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
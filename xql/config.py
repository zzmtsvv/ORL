from dataclasses import dataclass
import torch


@dataclass
class xql_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    state_dim: int = 17
    action_dim: int = 6

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_func_lr: float = 3e-4
    batch_size: int = 1024
    buffer_size: int = 1000000
    discount: float = 0.99
    hidden_dim: int = 256
    max_action: float = 1.0
    max_timesteps: int = int(1e6)
    tau: float = 5e-3

    value_update_freq: int = 1000
    beta: float = 1.0  # 10.0
    value_noise_std: float = 0.1
    exp_adv_temperature: float = 0.1
    advantage_max: float = 100.0
    critic_delta_loss: float = 20.0

    grad_clip: float = 7.0

    project: str = "XQL"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
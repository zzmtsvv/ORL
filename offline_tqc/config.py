from dataclasses import dataclass
import torch


@dataclass
class tqc_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    max_timesteps: int = int(1e6)

    state_dim: int = 17
    action_dim: int = 6

    batch_size: int = 256
    buffer_size: int = int(1e6)
    discount: float = 0.99
    tau: float = 5e-3

    max_action: float = 1.0

    num_quantiles: int = 25
    quantiles_to_drop_per_critic: int = 2
    num_critics: int = 5

    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    project: str = "TQC"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
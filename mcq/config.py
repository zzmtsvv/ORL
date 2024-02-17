from dataclasses import dataclass
import torch


@dataclass
class mcq_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    state_dim: int = 17
    action_dim: int = 6

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    vae_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    eta: float = 1.0
    gamma: float = 0.99
    hidden_dim: int = 256
    max_action: float = 1.0
    num_critics: int = 2
    max_timesteps: int = int(3e5)
    tau: float = 5e-3

    num_samples: int = 10
    lmbda: float = 0.7

    project: str = "MCQ"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
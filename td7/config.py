from dataclasses import dataclass
import torch


@dataclass
class td7_config:
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
    target_update_freq: int = 250
    exploration_noise: float = 0.1
    
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    max_action: float = 1.0

    alpha: float = 0.4
    min_priority: float = 1.0

    lambda_coef: float = 0.1

    embedding_dim: int = 256
    hidden_dim: int = 256
    encoder_lr: float = 3e-4
    encoder_activation: str = "elu"
    actor_lr: float = 3e-4
    actor_activation: str = "relu"
    critic_lr: float = 3e-4
    critic_activation: str = "elu"

    normalize_actions: bool = True
    priority_buffer: bool = True

    project: str = "TD7"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
from dataclasses import dataclass
import torch


@dataclass
class bppo_config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    state_dim: int = 17
    action_dim: int = 6

    value_lr: float = 1e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    batch_size: int = 512
    buffer_size: int = 1000000
    gamma: float = 0.99
    hidden_dim: int = 256
    max_action: float = 1.0
    tau: float = 5e-3
    target_update_freq: int = 2

    omega: float = 0.9
    clip_ratio: float = 0.25
    decay: float = 0.96
    entropy_weight: float = 0.01
    policy_grad_norm: float = 0.5

    lr_decay: bool = True
    clip_decay: bool = True

    value_steps: int = int(2e4)
    bc_steps: int = int(5e4)
    critic_steps: int = int(2e4)
    bppo_steps: int = int(1e4)

    # value_steps: int = int(2e6)
    # bc_steps: int = int(5e5)
    # critic_steps: int = int(2e6)
    # bppo_steps: int = int(1e3)

    project: str = "BPPO"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)

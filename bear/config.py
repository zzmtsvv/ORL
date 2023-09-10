from dataclasses import dataclass
import torch


@dataclass
class bear_config:
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
    lagrange_lr: float = 1e-3

    hidden_dim: int = 256
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005

    critic_ln: bool = True
    num_critics: int = 20
    normalize: bool = True

    mmd_kernel_type: str = "gaussian"  # laplacian
    mmd_sigma: float = 20.0
    critic_lambda: float = 0.75
    lagrange_threshold: float = 0.05

    num_warmup_iterations: int = 20_000

    project: str = "BEAR"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)
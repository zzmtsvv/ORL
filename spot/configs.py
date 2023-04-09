from dataclasses import dataclass
import os


@dataclass
class vae_config:
    seed: int = 0
    env: str = "hopper"  # halfcheetah walker2d
    dataset: str = "medium"  # medium, medium-replay, medium-expert, expert
    version: str = "v2"
    hidden_dim: int = 750
    beta: float = 0.5
    num_iterations: int = 100000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0
    use_scheduler: bool = False
    gamma: float = 0.95
    max_action_exists: bool = True
    clip_to_eps: bool = False
    eps: float = 1e-4
    #latent_dim: int  # action_dim * 2
    normalize_states: bool = True
    eval_size: float = 0.0
    weights_dir: str = "weights"
    base_dir: str = "spot"


@dataclass
class spot_config:
    save_video: bool = False
    buffer_size: int = 1000000
    env: str = "hopper"  # halfcheetah walker2d
    dataset: str = "medium"  # medium, medium-replay
    version: str = "v0"
    env_name: str = f"{env}-{dataset}-{version}"
    seed: int = 0
    eval_frequency: int = 5e3
    max_timesteps: int = 1000000
    save_model: bool = False
    save_final_model: bool = True
    eval_episodes: int = 10
    clip: bool = False
    exploration_noise: float = 0.1
    batch_size: int = 256
    discount_factor: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    lr: float = 3e-4
    actor_lr: float = None
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    actor_dropout: float = 0.1
    alpha: float = 0.4
    normalize_env: bool = True
    vae_model_path: str = os.path.join("spot", "weights", f"vae_{env}-{dataset}.pt")
    beta: float = 0.5
    use_importance_sampling: bool = False
    num_samples: int = 1
    lambda_: float = 1.0
    with_q_norm: bool = True
    lambda_cool: float = False
    lambda_end: float = 0.2
    base_dir: str = "spot"
    weights_dir: str = "policy_weights"

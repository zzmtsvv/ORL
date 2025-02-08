import os
import random
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

import torch
import pyrallis
import wandb

from replay_buffer import ReplayBuffer
from fql import FQL


@dataclass
class fql_config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    state_dim: int = 17
    action_dim: int = 6

    batch_size: int = 256

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    bc_alpha: float = 0.2
    num_flow_steps: int = 10
    num_critics: int = 2
    discount: float = 0.99
    hidden_dim: int = 256
    min_action: float = -1.0
    max_action: float = 1.0
    buffer_size: int = 1_000_000
    tau: float = 5e-3

    max_timesteps: int = 1_000_000

    project: str = "FQL"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["WANDB_ENTITY"] = "zzmtsvv"
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
os.environ["WANDB_API_KEY"] = "8cf7cafd958aa2df2d9f18fa32723dddc4024806"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@pyrallis.wrap()
def train_fql(config: fql_config):
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.deterministic = True
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed)

    dict_config = asdict(config)
    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=dict_config,
    )
    
    buffer = ReplayBuffer(
        config.state_dim,
        config.action_dim,
        config.buffer_size,
        DEVICE
    )
    buffer.from_json(config.dataset_name)

    fql = FQL(
        config.state_dim,
        config.action_dim,
        config.hidden_dim,
        config.bc_alpha,
        config.num_flow_steps,
        config.actor_lr,
        config.critic_lr,
        config.discount,
        config.tau,
        config.num_critics,
        config.min_action,
        config.max_action,
        DEVICE
    )

    print(f"Training starts on {config.device} 🚀")

    for t in tqdm(range(config.max_timesteps)):
        batch = buffer.sample(config.batch_size)

        logging_dict = fql.train_step(
            *batch,
            rng=rng
        )

        wandb.log(logging_dict, step=t + 1)


if __name__ == "__main__":
    train_fql()

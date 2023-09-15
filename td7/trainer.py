import os
import numpy as np
import torch
import random
from config import td7_config
from modules import TD7Actor, TD7Encoder, TD7Critic
from td7 import TD7
from replay_buffer import LAP

import wandb
from tqdm import tqdm


class TD7Trainer:
    def __init__(self,
                 cfg=td7_config) -> None:
        self.cfg = cfg
        seed = cfg.seed
        self.batch_size = cfg.batch_size
        self.device = cfg.device

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

        encoder = TD7Encoder(cfg.state_dim, cfg.action_dim, cfg.embedding_dim, cfg.hidden_dim)
        actor = TD7Actor(cfg.state_dim, cfg.action_dim, cfg.embedding_dim, cfg.hidden_dim)
        critic = TD7Critic(cfg.state_dim, cfg.action_dim, cfg.embedding_dim, cfg.hidden_dim)

        self.td7 = TD7(cfg, encoder, actor, critic)

        self.buffer = LAP(state_dim=cfg.state_dim,
                          action_dim=cfg.action_dim,
                          buffer_size=cfg.buffer_size,
                          max_action=cfg.max_action,
                          normalize_actions=cfg.normalize_actions,
                          with_priority=cfg.priority_buffer)
        self.buffer.from_json(cfg.dataset_name)

        self.buffer_device = self.buffer.device

    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="TD7 steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict, priority = self.td7.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                self.buffer.update_priority(priority.to(self.buffer_device))
                
                wandb.log(logging_dict, step=self.td7.total_iterations)

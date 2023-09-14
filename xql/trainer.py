import torch
from xql import XQL
from config import xql_config
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class XQLTrainer:
    def __init__(self,
                 cfg=xql_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        self.xql = XQL(cfg)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="XQL steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.xql.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.xql.total_iterations)
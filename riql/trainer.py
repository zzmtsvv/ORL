from dataclasses import asdict
import torch
from riql import RIQL
from config import riql_config
from modules import Actor, EnsembledCritic
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class RIQLTrainer:
    def __init__(self,
                 cfg=riql_config()) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        self.riql = RIQL(cfg)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        wandb.init(
            project=self.cfg.project,
            entity="zzmtsvv",
            group=self.cfg.group,
            name=self.cfg.name,
            config=asdict(self.cfg)
        )

        for t in tqdm(range(self.cfg.max_timesteps), desc="IQL steps"):
                
                # batch = self.buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

                # states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

            logging_dict = self.riql.train(states,
                                           actions,
                                           rewards,
                                           next_states,
                                           dones)
                
            wandb.log(logging_dict, step=self.riql.total_iterations)
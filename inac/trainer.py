import torch
from inac import InAC
from config import inac_config
from modules import Actor, EnsembledCritic, ValueFunction
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class InACTrainer:
    def __init__(self,
                 cfg=inac_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.batch_size = cfg.batch_size

        actor = Actor(cfg.state_dim,
                      cfg.action_dim,
                      cfg.hidden_dim)
        behavior = Actor(cfg.state_dim,
                      cfg.action_dim,
                      cfg.hidden_dim)
        critic = EnsembledCritic(cfg.state_dim,
                                 cfg.action_dim,
                                 cfg.hidden_dim)
        value = ValueFunction(cfg.state_dim,
                              cfg.hidden_dim)

        self.inac = InAC(cfg,
                         actor,
                         behavior,
                         critic,
                         value)
        
        self.buffer = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="InAC steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.inac.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.inac.total_iterations)
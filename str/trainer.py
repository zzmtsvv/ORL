import torch
from str import STR
from config import str_config
from modules import Actor, EnsembledCritic, ValueFunction
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class STRTrainer:
    def __init__(self,
                 cfg=str_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.batch_size = cfg.batch_size

        behavior = Actor(cfg.state_dim,
                      cfg.action_dim,
                      cfg.hidden_dim)
        critic = EnsembledCritic(cfg.state_dim,
                                 cfg.action_dim,
                                 cfg.hidden_dim,
                                 num_critics=cfg.num_critics)

        self.str = STR(cfg,
                         behavior,
                         critic)
        
        self.buffer = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
        self.buffer.normalize_states()
        self.buffer.clip()
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="Behavior Pretrain"):
                states, actions = [x.to(self.device) for x in self.buffer.sample(self.batch_size)[:2]]

                behavior_entropy = self.str.behavior_pretrain_step(states, actions)

                wandb.log({"behavior_entropy": behavior_entropy}, step=self.str.total_iterations)
            
            self.str.actor_init()

            for t in tqdm(range(self.cfg.max_timesteps), desc="STR steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.str.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.str.total_iterations)
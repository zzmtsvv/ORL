import wandb

from tqdm import tqdm
from config import bear_config
import random
import os
import numpy as np
import torch
from dataset import ReplayBuffer
from modules import Actor, EnsembledCritic
from bear import BEAR


class BEARTrainer:
    def __init__(self,
                 cfg=bear_config) -> None:
        self.cfg = cfg
        self.device = cfg.device
        seed = cfg.seed

        random.seed(seed)
        os.environ['PYTHONASSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        actor = Actor(self.state_dim, self.action_dim, cfg.hidden_dim, edac_init=True).to(self.device)

        critic = EnsembledCritic(self.state_dim, self.action_dim, cfg.hidden_dim, layer_norm=cfg.critic_ln, num_critics=cfg.num_critics).to(self.device)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)

        if cfg.normalize:
            _, _ = self.buffer.normalize_states()

        self.bear = BEAR(cfg,
                         actor,
                         critic)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")


        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="BEAR steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.bear.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.bear.total_iterations)
import random
import os
import numpy as np
from tqdm import tqdm
import torch
from config import prdc_config
from dataset import ReplayBuffer
from modules import DeterministicActor, EnsembledCritic
from prdc import PRDC

import wandb


class PRDCTrainer:
    def __init__(self,
                 cfg=prdc_config) -> None:
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

        actor = DeterministicActor(self.state_dim, self.action_dim, cfg.hidden_dim, edac_init=True).to(self.device)

        critic = EnsembledCritic(self.state_dim, self.action_dim, cfg.hidden_dim, layer_norm=cfg.critic_ln).to(self.device)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)

        if cfg.normalize:
            _, _ = self.buffer.normalize_states()
        
        states: np.ndarray = self.buffer.states.numpy()
        actions: np.ndarray = self.buffer.actions.numpy()

        kdtree_data = np.hstack([cfg.beta * states, actions])

        self.prdc = PRDC(cfg,
                         kdtree_data,
                         actor,
                         critic)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="PRDC steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.prdc.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.prdc.total_iterations)
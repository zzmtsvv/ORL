from dataclasses import asdict
import torch
from cross_q import CrossQ
from config import crossq_config
from modules import Actor, Critic
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class CrossQTrainer:
    def __init__(self,
                 cfg=crossq_config()) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        actor = Actor(self.state_dim, self.action_dim, cfg.hidden_dim, edac_init=True).to(self.device)

        critic1 = Critic(self.state_dim, self.action_dim, cfg.critic_hidden_dim).to(self.device)
        critic2 = Critic(self.state_dim, self.action_dim, cfg.critic_hidden_dim).to(self.device)

        self.cross_q = CrossQ(cfg,
                             actor,
                             critic1,
                             critic2)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        wandb.init(project=self.cfg.project,
                   entity="zzmtsvv",
                   group=self.cfg.group,
                   name=self.cfg.name,
                   config=asdict(self.cfg))

        for t in tqdm(range(self.cfg.max_timesteps), desc="CrossQ steps"):
            batch = self.buffer.sample(self.batch_size)

            states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

            logging_dict = self.cross_q.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
            wandb.log(logging_dict, step=self.cross_q.total_iterations)
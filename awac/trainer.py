import torch
from awac import AWAC
from config import awac_config
from modules import Actor, EnsembledCritic
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class AWACTrainer:
    def __init__(self,
                 cfg=awac_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        actor = Actor(self.state_dim, self.action_dim, cfg.hidden_dim).to(self.device)
        actor_optim = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr)

        critic = EnsembledCritic(self.state_dim, self.action_dim, cfg.hidden_dim).to(self.device)
        critic_optim = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)

        self.awac = AWAC(cfg,
                         actor,
                         actor_optim,
                         critic,
                         critic_optim)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="AWAC steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.awac.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.awac.total_iterations)
import torch
from fisher_brc import FisherBRC
from config import fbrc_config
from modules import Actor, EnsembledCritic, MixtureGaussianActor
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class FBRCTrainer:
    def __init__(self,
                 cfg=fbrc_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.batch_size = cfg.batch_size

        actor = Actor(cfg.state_dim,
                      cfg.action_dim,
                      cfg.hidden_dim)
        behavior = MixtureGaussianActor(cfg.state_dim,
                                        cfg.action_dim,
                                        cfg.hidden_dim,
                                        cfg.num_bc_actors)
        critic = EnsembledCritic(cfg.state_dim,
                                 cfg.action_dim,
                                 cfg.hidden_dim)

        self.fbrc = FisherBRC(cfg,
                         actor,
                         behavior,
                         critic)
        
        self.buffer = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=f"behavior_{self.cfg.group}", name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})
            
            for t in tqdm(range(self.cfg.behavior_pretrain_steps), desc="Behavior Pretrain"):
                batch = self.buffer.sample(self.batch_size)

                states, actions = [b.to(self.device) for b in batch[:2]]

                logging_dict = self.fbrc.behavior_pretrain(states, actions)

                wandb.log(logging_dict, step=self.fbrc.bc_pretrain_steps)
        
        wandb.finish()
            
        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=f"frbc_{self.cfg.group}", name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="FBRC steps"):
                    
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.fbrc.train(states,
                                               actions,
                                               rewards,
                                               next_states,
                                               dones)
                
                wandb.log(logging_dict, step=self.fbrc.total_iterations)
        
        wandb.finish()

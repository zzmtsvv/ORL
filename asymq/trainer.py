import torch
from asymq import AsymQ
from config import asymq_config
from modules import Actor, EnsembledCritic
from buffer import ReplayBuffer

import wandb
from tqdm import tqdm


class EDACTrainer:
    def __init__(self,
                 cfg=asymq_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        actor = Actor(self.state_dim, self.action_dim, cfg.hidden_dim, edac_init=True).to(self.device)

        critic = EnsembledCritic(self.state_dim, self.action_dim, cfg.hidden_dim).to(self.device)

        self.asymq = AsymQ(cfg,
                             actor,
                             critic)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="AsymQ steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.asymq.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.asymq.total_iterations)
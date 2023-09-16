import torch
from tqc import TQC
from config import tqc_config
from modules import Actor, TruncatedQuantileEnsembledCritic
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class TQCTrainer:
    def __init__(self,
                 cfg=tqc_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        actor = Actor(self.state_dim, self.action_dim, cfg.hidden_dim, edac_init=True).to(self.device)

        critic = TruncatedQuantileEnsembledCritic(self.state_dim,
                                                  self.action_dim,
                                                  cfg.num_quantiles,
                                                  cfg.hidden_dim,
                                                  cfg.num_critics).to(self.device)

        self.tqc = TQC(cfg,
                        actor,
                        critic)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="TQC steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.tqc.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.tqc.total_iterations)
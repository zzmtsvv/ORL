from dataclasses import asdict
import torch
from mcq import MCQ
from config import mcq_config
from modules import Actor, EnsembledCritic, ConditionalVAE
from buffer import ReplayBuffer

import wandb
from tqdm import tqdm


class MCQTrainer:
    def __init__(self,
                 cfg=mcq_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        actor = Actor(self.state_dim, self.action_dim, cfg.hidden_dim, edac_init=True).to(self.device)

        critic = EnsembledCritic(self.state_dim, self.action_dim, cfg.hidden_dim, num_critics=cfg.num_critics).to(self.device)

        vae = ConditionalVAE(self.state_dim, self.action_dim).to(self.device)

        self.mcq = MCQ(cfg,
                       actor,
                       critic,
                       vae)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        # wandb.init(
        #     project=self.cfg.project,
        #     entity="zzmtsvv",
        #     group=self.cfg.group,
        #     name=self.cfg.name,
        #     config=asdict(self.cfg)
        # )


        for t in tqdm(range(self.cfg.max_timesteps), desc="MCQ steps"):

            states, actions, rewards, next_states, next_actions = self.buffer.sample(self.batch_size)

            logging_dict = self.mcq.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 next_actions)
                
            # wandb.log(logging_dict, step=self.mcq.total_iterations)
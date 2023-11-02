from config import atac_config
from modules import StochasticActor, EnsembledCritic
from atac import ATAC
from buffer import ReplayBuffer
from tqdm import tqdm

import wandb


class ATACTrainer:
    def __init__(self,
                 cfg=atac_config) -> None:
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.device = cfg.device

        actor = StochasticActor(cfg.state_dim,
                                cfg.action_dim)
        critic = EnsembledCritic(cfg.state_dim,
                                 cfg.action_dim)
        
        self.buffer = ReplayBuffer(cfg.state_dim,
                                   cfg.action_dim,
                                   cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)

        value_max = max(0.0, self.buffer.rewards.max() / (1.0 - cfg.discount)).to(self.device)
        value_min = min(0.0, self.buffer.rewards.min() / (1.0 - cfg.discount), value_max - 1.0 / (1 - cfg.discount)).to(self.device)

        print(value_max, value_min)

        self.atac = ATAC(cfg,
                         actor,
                         critic,
                         value_max,
                         value_min)

    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        # with open("njf.txt", "w") as f:

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="ATAC steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.atac.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.atac.total_iterations)

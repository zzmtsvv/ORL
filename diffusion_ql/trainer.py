from dql import DiffusionQL
from config import dql_config
from dataset import ReplayBuffer

import wandb
from tqdm import tqdm


class DiffusionQLTrainer:
    def __init__(self,
                 cfg=dql_config) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.batch_size = cfg.batch_size

        self.buffer = ReplayBuffer(cfg.state_dim,
                                          cfg.action_dim,
                                          cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
        
        self.dql = DiffusionQL(cfg)

    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="DiffusionQL steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.dql.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.dql.total_iterations)

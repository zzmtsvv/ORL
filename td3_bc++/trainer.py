import torch
from config import td3bcplusplus_config
from td3_bc_plusplus import TD3BC_PlusPlus
from modules import DeterministicActor, EnsembledCritic
from dataset import ReplayBuffer
from tqdm import tqdm
import wandb


class TD3_BCPlusPlusTrainer:
    def __init__(self,
                 cfg=td3bcplusplus_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        actor = DeterministicActor(self.state_dim, self.action_dim, cfg.hidden_dim, edac_init=True).to(self.device)
        actor_optim = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_learning_rate)

        critic = EnsembledCritic(self.state_dim, self.action_dim, cfg.hidden_dim).to(self.device)
        critic_optim = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_learning_rate)

        self.td3 = TD3BC_PlusPlus(cfg,
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

            for t in tqdm(range(self.cfg.max_timesteps), desc="TD3 BC++ steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

                logging_dict = self.td3.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 dones)
                
                wandb.log(logging_dict, step=self.td3.total_iterations)

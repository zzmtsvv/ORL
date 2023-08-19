import torch
from config import rebrac_config
from rebrac import ReBRAC
from modules import DeterministicActor, EnsembledCritic
from dataset import ReplayBuffer
from tqdm import tqdm
import wandb


class ReBRACTrainer:
    def __init__(self,
                 cfg=rebrac_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.state_dim = 17
        self.action_dim = 6
        self.batch_size = cfg.batch_size

        actor = DeterministicActor(self.state_dim, self.action_dim, cfg.hidden_dim, edac_init=True).to(self.device)
        actor_optim = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_learning_rate)

        critic = EnsembledCritic(self.state_dim, self.action_dim, cfg.hidden_dim).to(self.device)
        critic_optim = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_learning_rate)

        self.rebrac = ReBRAC(cfg,
                             actor,
                             actor_optim,
                             critic,
                             critic_optim)
        
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, 3, cfg.buffer_size)
        self.buffer.from_json(cfg.dataset_name)
    
    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for t in tqdm(range(self.cfg.max_timesteps), desc="ReBRAC steps"):
                
                batch = self.buffer.sample(self.batch_size)

                states, actions, rewards, next_states, next_actions = [x.to(self.device) for x in batch[:-1]]

                logging_dict = self.rebrac.train(states,
                                                 actions,
                                                 rewards,
                                                 next_states,
                                                 next_actions)
                
                wandb.log(logging_dict, step=self.rebrac.total_iterations)

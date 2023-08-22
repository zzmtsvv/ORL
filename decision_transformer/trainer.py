from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import SequenceDataset
from utils import seed_everything, load_trajectories
from config import dt_config
from model import DecisionTransformer
from loss import DTLoss

import wandb


class DTrainer:
    def __init__(self,
                 cfg=dt_config) -> None:
        self.cfg = cfg
        self.device = cfg.device
        
        seed_everything(cfg.seed)

        dataset, info = load_trajectories(cfg.dataset_name)
        dataset = SequenceDataset(dataset,
                                  info,
                                  cfg.sequence_length,
                                  cfg.reward_scale)
        
        trainloader = DataLoader(dataset,
                                 batch_size=cfg.batch_size,
                                 pin_memory=True,
                                 num_workers=cfg.num_workers)
        
        self.trainloader_iter = iter(trainloader)
        
        self.model = DecisionTransformer(cfg,
                                         cfg.state_dim,
                                         cfg.action_dim).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=cfg.learning_rate,
                                           weight_decay=cfg.weight_decay,
                                           betas=cfg.betas)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps + 1) / cfg.warmup_steps, 1))
        
        self.loss = DTLoss()
        
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def fit(self):
        print(f"Training starts on {self.device}🚀")

        with wandb.init(project=self.cfg.project, entity="zzmtsvv", name=self.cfg.name, group=self.cfg.group):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            for step in tqdm(range(self.cfg.training_steps), desc="DT Training"):
                batch = next(self.trainloader_iter)

                states, actions, returns, time_steps, mask = [b.to(self.device) for b in batch]

                padding_mask = ~mask.to(torch.bool)
                padding_mask = padding_mask.to(self.device)

                predicted_actions = self.model(states=states,
                                               actions=actions,
                                               mc_returns=returns,
                                               time_steps=time_steps,
                                               key_padding_mask=padding_mask)
                
                loss = self.loss(predicted_actions, actions, mask)

                self.optimizer.zero_grad()
                loss.backward()

                if self.cfg.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()

                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                }, step=step)

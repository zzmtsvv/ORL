from typing import Dict
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from config import dql_config
from actor_modules import MLP, DDPM
from critic_modules import EnsembledCritic


class DiffusionQL:
    def __init__(self,
                 cfg: dql_config) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.discount = cfg.discount
        self.grad_norm = cfg.grad_norm
        self.eta = cfg.eta
        self.tau = cfg.tau

        diffusion_trunk = MLP(cfg.state_dim, cfg.action_dim, cfg.hidden_dim)
        self.actor = DDPM(cfg.state_dim,
                          cfg.action_dim,
                          diffusion_trunk,
                          cfg.max_action,
                          cfg.T).to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.learning_rate)
        self.actor_target = deepcopy(self.actor)

        self.critic = EnsembledCritic(cfg.state_dim,
                                      cfg.action_dim,
                                      cfg.hidden_dim).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.learning_rate)
        self.critic_target = deepcopy(self.critic)

        self.actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim,
                                                                             T_max=cfg.max_timesteps // 1000,
                                                                             eta_min=0.0)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optim,
                                                                              T_max=cfg.max_timesteps // 1000,
                                                                              eta_min=0.0)
        
        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        # critic step
        current_q = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            
            tgt_q = self.critic(next_states, next_actions).min(0).values

            tgt_q = rewards + self.discount * (1.0 - dones) * tgt_q.unsqueeze(-1)
        
        critic_loss = F.mse_loss(current_q, tgt_q.squeeze(-1))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(),
                                                    max_norm=self.grad_norm,
                                                    norm_type=2)
        self.critic_optim.step()

        # actor step
        bc_loss = self.actor.loss(actions, states)
        pi = self.actor(states)

        q_values = self.critic(states, pi)

        # taken from original implementation - pretty interesting trick
        if np.random.uniform() > 0.5:
            numerator_index, denominator_index = 0, 1
        else:
            numerator_index, denominator_index = 1, 0
        
        q_loss = -q_values[numerator_index].mean() / q_values[denominator_index].abs().mean().detach()

        actor_loss = bc_loss + self.eta * q_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                   max_norm=self.grad_norm,
                                                   norm_type=2)
        self.actor_optim.step()

        if not self.total_iterations % self.cfg.actor_update_freq:
            self.soft_actor_update()
        
        self.soft_critic_update()

        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            "actor_loss": actor_loss.item(),
            "bc_loss": bc_loss.item(),
            "q_loss": q_loss.item(),
            "critic_loss": critic_loss.item(),
            "target_q_mean": tgt_q.mean().item(),
            "actor_grad_norm": actor_grad_norm.max().item(),
            "critic_grad_norm": critic_grad_norm.max().item(),
            "learning_rate": self.actor_lr_scheduler.get_last_lr()[0]
        }

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
    
    def soft_actor_update(self):
        if self.total_iterations < self.cfg.steps_not_updating_actor_target:
            return
        
        for param, tgt_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

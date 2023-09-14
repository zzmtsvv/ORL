from typing import Dict
from copy import deepcopy
import torch
from torch.nn import functional as F
from config import xql_config
from modules import Actor, EnsembledCritic, ValueFunction


class XQL:
    def __init__(self,
                 cfg: xql_config) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.actor = Actor(cfg.state_dim,
                           cfg.action_dim,
                           cfg.hidden_dim).to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim, cfg.max_timesteps)

        self.critic = EnsembledCritic(cfg.state_dim,
                                      cfg.action_dim,
                                      cfg.hidden_dim).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        
        with torch.no_grad():
            self.critic_target = deepcopy(self.critic).to(self.device)
        
        self.value_func = ValueFunction(cfg.state_dim,
                                        cfg.hidden_dim).to(self.device)
        self.value_optim = torch.optim.AdamW(self.value_func.parameters(), lr=cfg.value_func_lr)

        self.tau = cfg.tau
        self.discount = cfg.discount
        self.value_update_freq = cfg.value_update_freq
        self.noise_std = cfg.value_noise_std
        self.max_action = cfg.max_action
        self.beta = cfg.beta
        self.max_clip = cfg.grad_clip
        self.exp_adv_temperature = cfg.exp_adv_temperature
        self.advantage_max = cfg.advantage_max
        self.critic_delta_loss = cfg.critic_delta_loss

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1
        logging_dict = dict()

        # value func step
        if not self.total_iterations % self.value_update_freq:
            noise = (torch.randn_like(actions) * self.noise_std).clamp(-self.max_action / 2, self.max_action / 2)
            noised_actions = (actions + noise).clamp(-self.max_action, self.max_action)

            with torch.no_grad():
                tgt_q = self.critic_target(states, noised_actions).min(0).values
                
            value = self.value_func(states)

            value_loss = self.gumbel_loss(value, tgt_q, self.beta, self.max_clip)
                
            logging_dict["value_loss"] = value_loss.item()
                
            clip_ratio = (((tgt_q - value) / self.beta) > self.max_clip).float().mean()
            logging_dict["clip_ratio"] = clip_ratio.item()

            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

        # actor step
        with torch.no_grad():
            value = self.value_func(states)
            tgt_q = self.critic_target(states, actions).min(0).values
        
        exp_advantage = torch.exp((tgt_q - value.detach()) * self.exp_adv_temperature).clamp_max(self.advantage_max)
        bc_losses = -self.actor.log_prob(states, actions)
        actor_loss = (bc_losses * exp_advantage).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.lr_scheduler.step()

        logging_dict["actor_loss"] = actor_loss.item()
        logging_dict["actor_learning_rate"] = self.lr_scheduler.get_last_lr()[0]

        # critic step
        with torch.no_grad():
            next_v = self.value_func(next_states)
            v_target = self.critic_target(states, actions).min(0).values

            q_target = rewards + (1.0 - dones) * self.discount * next_v
        
        current_q = self.critic(states, actions)
        critic_loss = 2 * F.huber_loss(current_q, q_target.squeeze(-1), delta=self.critic_delta_loss)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        logging_dict["critic_loss"] = critic_loss.item()

        self.soft_critic_update()

        return logging_dict
    
    @staticmethod
    def gumbel_loss(prediction: torch.Tensor,
                    target: torch.Tensor,
                    beta: float = 1.0,
                    clip: float = 7.0) -> torch.Tensor:
        z = (target - prediction) / beta
        z = z.clamp(-clip, clip)
        z_max = z.max(dim=0).values
        z_max = torch.where(z_max < -1.0, -1.0, z_max)
        z_max = z_max.detach()

        loss = torch.exp(z - z_max) - z * torch.exp(-z_max) - torch.exp(-z_max)
        return loss.mean()
    
    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

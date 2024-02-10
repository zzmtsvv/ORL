from typing import Dict, Tuple
from copy import deepcopy
import torch
from torch.nn import functional as F
from modules import Actor, EnsembledCritic
from config import str_config


class STR:
    def __init__(self,
                 cfg: str_config,
                 behavior: Actor,
                 critic: EnsembledCritic) -> None:
        self.cfg = cfg

        self.device = cfg.device
        self.discount = cfg.discount
        self.temperature = cfg.temperature
        self.tau = cfg.tau

        # self.actor = actor.to(self.device)
        # self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.behavior = behavior.to(self.device)
        self.behavior_optim = torch.optim.Adam(self.behavior.parameters(), lr=cfg.behavior_lr)
        
        self.critic = critic.to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.critic_target = deepcopy(critic).to(self.device)

        self.adv_max = cfg.advantage_max
        self.adv_min = cfg.advantage_min

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_states)
            
            target_q = self.critic_target(next_states, next_actions).min(0).values
            target_q = target_q.unsqueeze(-1) - self.temperature * next_log_prob

            target_q = rewards + self.discount * (1.0 - dones) * target_q
        
        current_q = self.critic(states, actions)
        qq = current_q.clone().detach()

        critic_loss = F.mse_loss(current_q, target_q.squeeze(-1))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor update
        behavior_log_prob = self.behavior.log_prob(states, actions).detach().squeeze(1)
        actor_log_prob = self.actor.log_prob(states, actions).squeeze(1)

        importance_ratio = actor_log_prob.detach().exp() / (behavior_log_prob.exp() + 1e-3)

        actor_mean = self.actor.get_policy(states).mean
        advantage = qq - self.critic(states, actor_mean)
        
        actor_objective = importance_ratio * torch.exp(advantage / self.temperature).clamp_max(self.adv_max) * actor_log_prob
        actor_loss = -actor_objective.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.lr_scheduler.step()

        self.soft_critic_update()
        
        return {
            "critic_loss": critic_loss.item(),
            "current_q": qq.mean().item(),
            "actor_loss": actor_loss.item(),
            "actor_learning_rate": self.lr_scheduler.get_last_lr()[0]
        }

    def behavior_pretrain_step(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        loss = -self.behavior.log_prob(states, actions).mean()
        self.total_iterations += 1

        self.behavior_optim.zero_grad()
        loss.backward()
        self.behavior_optim.step()

        return loss.item()
    
    def actor_init(self) -> None:
        self.actor = deepcopy(self.behavior).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim, self.cfg.max_timesteps)

        self.total_iterations = 0

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

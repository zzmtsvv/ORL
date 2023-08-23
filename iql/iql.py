from typing import Dict, Union
from copy import deepcopy
import torch
from torch.nn import functional as F
from config import iql_config
from modules import Actor, ValueFunction, EnsembledCritic


_Number = Union[float, int]


class IQL:
    def __init__(self,
                 cfg: iql_config) -> None:
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

        self.iql_tau = cfg.iql_tau
        self.tau = cfg.tau
        self.discount = cfg.discount
        self.beta = cfg.beta
        self.exp_adv_max = cfg.exp_adv_max

        self.total_iterations = 0
    
    def assymetric_l2(self, u: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(self.iql_tau - (u < 0).float()) * u.pow(2)

        return loss.mean()
    
    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, _Number]:
        
        self.total_iterations += 1

        with torch.no_grad():
            next_v = self.value_func(next_states)
            v_target = self.critic_target(states, actions).min(0).values

            q_target = rewards + (1.0 - dones) * self.discount * next_v
        
        # value func step
        value = self.value_func(states)
        advantage = v_target - value
        value_loss = self.assymetric_l2(advantage)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        # critic step
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, q_target.squeeze(-1))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor step
        exp_advantage = torch.exp(self.beta * advantage.detach()).clamp_max(self.exp_adv_max)

        bc_losses = -self.actor.log_prob(states, actions)
        actor_loss = (bc_losses * exp_advantage).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.lr_scheduler.step()

        return {
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_learning_rate": self.lr_scheduler.get_last_lr()[0]
        }

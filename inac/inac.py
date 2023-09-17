from typing import Tuple, Dict
import torch
from torch.nn import functional as F
from copy import deepcopy
from config import inac_config
from modules import Actor, EnsembledCritic, ValueFunction


class InAC:
    def __init__(self,
                 cfg: inac_config,
                 actor: Actor,
                 behavior: Actor,
                 critic: EnsembledCritic,
                 value_func: ValueFunction) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.discount = cfg.discount
        self.temperature = cfg.temperature
        self.tau = cfg.tau

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.behavior = behavior.to(self.device)
        self.behavior_optim = torch.optim.AdamW(self.behavior.parameters(), lr=cfg.behavior_lr)
        
        self.critic = critic.to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        self.critic_target = deepcopy(critic).to(self.device)

        self.value_func = value_func.to(self.device)
        self.value_optim = torch.optim.AdamW(self.value_func.parameters(), lr=cfg.value_func_lr)

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

        # behavior step
        behavior_loss = self.behavior_loss(states, actions)

        self.behavior_optim.zero_grad()
        behavior_loss.backward()
        self.behavior_optim.step()

        # value func step
        value_loss = self.value_loss(states)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        # critic step
        critic_loss, q_values = self.critic_loss(states, actions, rewards, next_states, dones)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor step
        actor_loss, actor_entropy = self.actor_loss(states, actions)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_critic_update()

        return {
            "behavior_entropy": behavior_loss.item(),
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_values": q_values,
            "actor_loss": actor_loss.item(),
            "actor_entropy": actor_entropy
        }

    def behavior_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        loss = -self.behavior.log_prob(states, actions).mean()
        return  loss
    
    def value_loss(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions, log_prob = self.actor(states)

            target = self.critic_target(states, actions).min(0).values
            target = target.unsqueeze(-1) - self.temperature * log_prob
        
        value = self.value_func(states)
        loss = F.mse_loss(value, target)
        return loss
    
    def critic_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_states: torch.Tensor,
                    dones: torch.Tensor) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_states)
            
            target_q = self.critic_target(next_states, next_actions).min(0).values
            target_q = target_q.unsqueeze(-1) - self.temperature * next_log_prob

            target_q = rewards + self.discount * (1.0 - dones) * target_q
        
        current_q = self.critic(states, actions)

        loss = F.mse_loss(current_q, target_q.squeeze(-1))
        return loss, current_q.mean().item()
    
    def actor_loss(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, float]:
        log_prob = self.actor.log_prob(states, actions)

        with torch.no_grad():
            min_q = self.critic(states, actions).min(0).values
            value = self.value_func(states)
            
            behavior_log_prob = self.behavior.log_prob(states, actions)
            advantage = torch.exp((min_q - value) / self.temperature - behavior_log_prob).clip(self.adv_min, self.adv_max)
        
        loss = -(advantage * log_prob).mean()
        return loss, -log_prob.mean().item()

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

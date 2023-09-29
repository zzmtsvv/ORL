from typing import Dict
from copy import deepcopy
import torch
from torch.nn import functional as F
from config import eql_config
from modules import Actor, EnsembledCritic, ValueFunction

class EQL:
    def __init__(self,
                 cfg: eql_config,
                 actor: Actor,
                 critic: EnsembledCritic,
                 value_func: ValueFunction) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.discount = cfg.discount
        self.alpha = cfg.alpha

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim, cfg.max_timesteps)
        
        self.critic = critic.to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        with torch.no_grad():
            self.critic_target = deepcopy(critic).to(self.device)

        self.value_func = value_func.to(self.device)
        self.value_optim = torch.optim.AdamW(self.value_func.parameters(), lr=cfg.value_func_lr)

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        target_current_q = self.critic_target(states, actions)
        tgt_q_min = target_current_q.min(0).values.detach()

        # value step
        current_v = self.value_func(states)

        advantage = ((tgt_q_min - current_v) / self.alpha).clamp_max(5.0)  # taken from original realization
        max_adv = torch.max(advantage, dim=0).values.detach().clamp_min(-1.0)

        value_loss = (torch.exp(advantage - max_adv) + torch.exp(-max_adv) * current_v / self.alpha).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        # actor step
        weights = torch.exp(self.cfg.beta * (tgt_q_min - current_v.detach()) / self.alpha)
        weights = weights.clamp(0.0, self.cfg.adv_max)

        log_probs = self.actor.log_prob(states, actions)
        actor_loss = -(weights * log_probs).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.lr_scheduler.step()

        # critic step
        with torch.no_grad():
            next_v = self.value_func(next_states)

            tgt_q = rewards + self.discount * (1.0 - dones) * next_v
        
        current_q = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_q, tgt_q.squeeze(-1))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.soft_critic_update()
        
        return {
            "value_loss": value_loss.item(),
            "value": current_v.mean().item(),
            "advantage_mean": (tgt_q_min - current_v.detach()).mean().item(),
            "actor_loss": actor_loss.item(),
            "actor_entropy": -log_probs.mean().item(),
            "learning_rate": self.lr_scheduler.get_last_lr()[0],
            "q_values": current_q.mean().item(),
            "critic_loss": critic_loss.item()
        }

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

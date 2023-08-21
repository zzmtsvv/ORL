from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from modules import Actor, EnsembledCritic
from config import awac_config
from typing import Dict, Union, Any


_Number = Union[float, int]


class AWAC:
    def __init__(self,
                 cfg: awac_config,
                 actor: Actor,
                 actor_optimizer: torch.optim.Optimizer,
                 critic: EnsembledCritic,
                 critic_optimizer: torch.optim.Optimizer) -> None:
        
        self.cfg = cfg
        self.device = cfg.device
        
        self.actor = actor.to(self.device)
        self.actor_optimizer = actor_optimizer

        self.critic = critic.to(self.device)
        self.critic_optimizer = critic_optimizer
        self.target_critic = deepcopy(critic).to(self.device)

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.awac_lambda = cfg.awac_lambda
        self.exp_adv_max = cfg.exp_adv_max

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, _Number]:
        self.total_iterations += 1

        # critic step
        with torch.no_grad():
            next_action, _ = self.actor(next_states)

            q_next = self.target_critic(next_states, next_action).min(0).values

            assert q_next.unsqueeze(-1).shape == dones.shape == rewards.shape
            tgt_q = rewards + self.gamma * (1 - dones) * q_next.unsqueeze(-1)
        
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, tgt_q.squeeze(1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor step
        with torch.no_grad():
            policy_action, _ = self.actor(states)
            q_pi = self.critic(states, policy_action)
            value = q_pi.min(0).values

            q_beta = self.critic(states, actions)
            q_value = q_beta.min(0).values

            advantage = q_value - value
            weights = torch.clamp_max(
                torch.exp(advantage / self.awac_lambda), self.exp_adv_max
            )
        action_log_prob = self.actor.log_prob(states, actions)
        actor_loss = (-action_log_prob * weights).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_critic_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }

    def actor_loss(self,
                   states: torch.Tensor,
                   actions: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            policy_action, _ = self.actor(states)
            policy_action = policy_action.detach()
            q_pi = self.critic(states, policy_action)
            value = q_pi.min(0).values

            q_beta = self.critic(states, actions)
            q_value = q_beta.min(0).values

            advantage = q_value - value
            weights = torch.clamp_max(
                torch.exp(advantage / self.awac_lambda), self.exp_adv_max
            )
        action_log_prob = self.actor.log_prob(states, actions)
        loss = -action_log_prob * weights
        return loss.mean()

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict()
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])

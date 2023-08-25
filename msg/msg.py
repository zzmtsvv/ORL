from typing import Dict, Union, Tuple
from copy import deepcopy
import torch
from torch.nn import functional as F
from modules import Actor, EnsembledCritic
from config import msg_config


_Number = Union[float, int]


class MSG:
    def __init__(self,
                 cfg: msg_config,
                 actor: Actor,
                 critic: EnsembledCritic,
                 alpha_lr: float = 3e-4) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.gamma = cfg.gamma
        self.beta = cfg.beta

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = critic.to(self.device)
        with torch.no_grad():
            self.critic_target = deepcopy(critic).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        self.target_entropy = -float(actor.action_dim)
        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)

        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, _Number]:
        self.total_iterations += 1

        alpha_loss = self.alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        actor_loss, batch_entropy, q_policy_std, q_policy_min = self.actor_loss(states)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        critic_loss = self.critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        with torch.no_grad():
            self.soft_critic_update()

            max_action = self.actor.max_action
            random_action = -max_action + 2 * max_action * torch.rand_like(actions)
            q_random = self.critic(states, random_action)
            q_random_std = q_random.std(0).mean().item()
            q_random_min = q_random.min(0).values.mean().item()
        
        return {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_policy_min": q_policy_min,
            "q_random_std": q_random_std,
            "q_random_min": q_random_min
        }

    def alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, log_prob = self.actor(state, need_log_prob=True)
        
        loss = -self.log_alpha * (log_prob + self.target_entropy)
        return loss.mean()
    
    def actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float, float]:
        action, log_prob = self.actor(state, need_log_prob=True)
        q_values = self.critic(state, action)

        assert q_values.shape[0] == self.critic.num_critics

        q_value_min = q_values.min(0).values.mean().item()
        q_value_std = q_values.std(0).mean().item()
        batch_entropy = -log_prob.mean().item()

        loss = self.alpha * log_prob - (q_values.mean(dim=0) + self.beta * q_values.std(dim=0))
        return loss.mean(), batch_entropy, q_value_std, q_value_min
    
    def critic_loss(self,
                    state: torch.Tensor,
                    action: torch.Tensor,
                    reward: torch.Tensor,
                    next_state: torch.Tensor,
                    done: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)
            q_next = self.critic_target(next_state, next_action)
            q_next = q_next - self.alpha * next_action_log_prob

            # assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward.squeeze(1) + self.gamma * (1 - done.squeeze(1)) * q_next
        
        q_values = self.critic(state, action)
        critic_loss = F.mse_loss(q_values,  q_target)

        return critic_loss
    
    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

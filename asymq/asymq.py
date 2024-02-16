from math import log
import torch
from torch.nn import functional as F
from copy import deepcopy
from config import asymq_config
from modules import Actor, EnsembledCritic
from typing import Dict, Any, Union, Tuple


_Number = Union[float, int]


class AsymQ:
    def __init__(self,
                 cfg: asymq_config,
                 actor: Actor,
                 critic: EnsembledCritic,
                 alpha_lr: float = 3e-4) -> None:
        
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.gamma = cfg.gamma
        
        self.q_temperature = cfg.q_temperature
        self.beta_up = cfg.beta_up
        self.beta_down = cfg.beta_down
        self.beta_multiplier = cfg.beta_multiplier

        self.target_entropy = -float(actor.action_dim)

        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)

        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.actor = actor.to(self.device)
        self.actor_target = deepcopy(actor).to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = critic.to(self.device)
        self.critic_target = deepcopy(critic).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

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

        actor_loss, batch_entropy, q_policy_std = self.actor_loss(states)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        critic_loss, weights = self.critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if not self.total_iterations % self.cfg.temperature_change_freq:
            effective_batch_ratio = (1 / (weights.pow(2).sum() + 1e-7) / states.shape[0]).item()

            if effective_batch_ratio >= self.beta_up:
                self.q_temperature *= self.beta_multiplier
            if effective_batch_ratio <= self.beta_down:
                self.q_temperature /= self.beta_multiplier

        with torch.no_grad():
            self.soft_critic_update()

            max_action = self.actor.max_action
            random_action = -max_action + 2 * max_action * torch.rand_like(actions)
            q_random_std = self.critic(states, random_action).std(0).mean().item()
        
        return {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }

    def alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, log_prob = self.actor(state, need_log_prob=True)
        
        loss = -self.log_alpha * (log_prob + self.target_entropy)
        return loss.mean()
    
    def actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, log_prob = self.actor(state, need_log_prob=True)
        q_values = self.critic(state, action)

        assert q_values.shape[0] == self.critic.num_critics

        q_value_min = q_values.min(0).values
        q_value_std = q_values.std(0).mean().item()
        batch_entropy = -log_prob.mean().item()

        assert log_prob.shape == q_value_min.shape
        loss = self.alpha * log_prob - q_value_min

        return loss.mean(), batch_entropy, q_value_std

    def critic_loss(self,
                    state: torch.Tensor,
                    action: torch.Tensor,
                    reward: torch.Tensor,
                    next_state: torch.Tensor,
                    done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)
            q_next = self.critic_target(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)
        
        q_values = self.critic(state, action)
        td_error = q_target.squeeze(1) - q_values
        errors = td_error.pow(2)

        with torch.no_grad():
            weight_logits = torch.clip(-td_error / self.q_temperature, -log(2.0), log(2.0))
            weights = torch.softmax(weight_logits) * batch_size
        
        loss = (weights * errors).mean()
        return loss, weights
      
    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["target_critic"])
        self.actor_optim.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optim.load_state_dict(state_dict["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()

from copy import deepcopy
import torch
from torch.nn import functional as F
from config import bppo_config
from modules import ValueFunction, Critic, StochasticActor, log_prob


eps = 1e-10


class BPPO:
    def __init__(self,
                 cfg: bppo_config) -> None:
        self.device = cfg.device
        self.omega = cfg.omega
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.clip_ratio = cfg.clip_ratio
        self.decay = cfg.decay
        self.lr_decay = cfg.lr_decay
        self.clip_decay = cfg.clip_decay
        self.entropy_weight = cfg.entropy_weight
        self.policy_grad_norm = cfg.policy_grad_norm

        self.value_func = ValueFunction(cfg.state_dim,
                                        cfg.hidden_dim).to(self.device)
        self.value_optim = torch.optim.Adam(self.value_func.parameters(), lr=cfg.value_lr)

        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        with torch.no_grad():
            self.critic_target = deepcopy(self.critic).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.policy = StochasticActor(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.actor_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optim,
            step_size=2,
            gamma=0.98
        )

        self.old_policy = deepcopy(self.policy).to(self.device)
    
    def policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        old_distribution = self.old_policy.get_policy(states)
        actions = old_distribution.rsample()

        advantages: torch.Tensor = self.critic(states, actions) - self.value_func(states)
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        new_distribution = self.policy.get_policy(states)

        new_log_prob = log_prob(new_distribution, actions)
        old_log_prob = log_prob(old_distribution, actions)
        ratio = (new_log_prob - old_log_prob).exp()

        advantages = self.weighted_advantage(advantages)

        loss1 = ratio * advantages

        if self.clip_decay:
            self.clip_ratio *= self.decay

        loss2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

        entropy_loss = new_distribution.entropy().sum(-1, keepdim=True) * self.entropy_weight

        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()
        return loss
    
    def policy_update(self, states: torch.Tensor) -> float:
        loss = self.policy_loss(states)

        self.policy_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_grad_norm)
        self.policy_optim.step()

        if self.lr_decay:
            self.scheduler.step()
        
        return loss.item()
    
    def value_update(self,
                     states: torch.Tensor,
                     returns: torch.Tensor) -> float:
        value_loss = F.mse_loss(self.value_func(states), returns)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        return value_loss.item()
    
    def critic_update(self,
                      states: torch.Tensor,
                      actions: torch.Tensor,
                      rewards: torch.Tensor,
                      next_states: torch.Tensor,
                      next_actions: torch.Tensor,
                      dones: torch.Tensor) -> float:
        with torch.no_grad():
            tgt_q = rewards + (1.0 - dones) * self.gamma * self.critic_target(next_states, next_actions)
        
        q_values = self.critic(states, actions)
        loss = F.mse_loss(q_values, tgt_q)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.item()

    def weighted_advantage(self, advantages: torch.Tensor) -> torch.Tensor:
        if self.omega == 0.5:
            return advantages
        
        weights = torch.zeros_like(advantages)
        indexes = torch.where(advantages > 0)[0]
        weights[indexes] = self.omega
        weights[torch.where(weights == 0)[0]] = 1.0 - self.omega
        weights = weights.to(self.device)

        return weights * advantages

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def behavior_cloning_update(self,
                                states: torch.Tensor,
                                actions: torch.Tensor) -> float:
        log_prob = self.policy.log_prob(states, actions)

        loss = (-log_prob).mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        return loss.item()

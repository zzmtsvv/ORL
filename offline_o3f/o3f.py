from typing import Dict, Tuple
from copy import deepcopy
import torch
from torch.nn import functional as F
from config import o3f_config
from modules import Actor, EnsembledCritic


class O3F:
    def __init__(self,
                 cfg: o3f_config,
                 actor: Actor,
                 crtiic: EnsembledCritic) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.discount = cfg.discount
        self.tau = cfg.tau
        self.std = cfg.standard_deviation
        self.action_candidates_num = cfg.num_action_candidates

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = crtiic.to(self.device)
        self.critic_target = deepcopy(crtiic).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        self.target_entropy = -float(cfg.max_action)
        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=cfg.alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
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

        critic_loss = self.critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.soft_critic_update()
        with torch.no_grad():

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
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_action_log_prob = self.actor(next_states, need_log_prob=True)

            # sample from N(next_actions, config.standard_deviation)
            standard_normal = torch.randn(self.action_candidates_num, *next_actions.shape).to(states.device)

            action_candidates = self.std * standard_normal + next_actions  # [num_candidates, batch_size, action_dim]
            action_candidates = action_candidates.view(-1, self.actor.action_dim)

            # [num_critics, num_candidates x batch_size, 1]
            q_candidates = self.critic_target(next_states.repeat_interleave(self.action_candidates_num, dim=0), action_candidates)
            q_candidates = q_candidates.view(self.cfg.num_critics, self.action_candidates_num, -1, 1)

            # [num_candidates, batch_size, 1]
            q_candidates_mean = q_candidates.mean(dim=0)
            candidates_indexes = torch.argmax(q_candidates_mean, dim=0).squeeze(-1)
            # print(candidates_indexes)

            q_candidates = q_candidates.view(self.cfg.num_critics, -1, 1)
            q_target = q_candidates.min(0).values[candidates_indexes].squeeze(-1)
            q_target = q_target - self.alpha * next_action_log_prob
            q_target = rewards + self.discount * (1.0 - dones) * q_target.unsqueeze(-1)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, q_target.squeeze(1))

        return critic_loss
    
    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

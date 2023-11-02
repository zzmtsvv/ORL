from typing import Dict
from copy import deepcopy
from config import atac_config
import torch
from torch.nn import functional as F
from modules import StochasticActor, EnsembledCritic


class ATAC:
    def __init__(self,
                 cfg: atac_config,
                 actor: StochasticActor,
                 critic: EnsembledCritic,
                 value_max: float,
                 value_min: float) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.discount = cfg.discount
        self.omega = cfg.omega
        self.constraint = cfg.weight_norm_constraint
        self.beta = cfg.beta

        self.value_max = value_max
        self.value_min = value_min

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.slow_lr)

        self.critic = critic.to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.fast_lr)

        self.critic_target = deepcopy(self.critic).requires_grad_(False)

        self.target_entropy = -float(actor.action_dim)
        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.fast_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        # critic step
        with torch.no_grad():
            next_actions, _ = self.actor(next_states)
            target_q = self.critic_target(next_states, next_actions).min(0).values
            target_q = rewards + (1.0 - dones) * self.discount * target_q.unsqueeze(-1)
            target_q.clamp_(self.value_min, self.value_max)

        policy = self.actor.get_policy(states)
        pi = policy.rsample()

        current_q = self.critic(states, actions)
        next_q = self.critic(states, actions)
        new_q = self.critic(states, pi.detach())

        target_error = F.mse_loss(current_q, target_q.squeeze(-1))
        q_backup = (rewards + (1.0 - dones) * self.discount * next_q.unsqueeze(-1)).clamp(self.value_min, self.value_max)
        residual_error = F.mse_loss(current_q, q_backup.squeeze(-1))
        
        bellman_error = (1.0 - self.omega) * target_error + self.omega * residual_error
        pessimism_loss = (new_q - current_q).mean()
        
        critic_loss = self.beta * bellman_error + pessimism_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.critic.apply(self.l2_projection(self.constraint))

        # actor step
        log_pi = policy.log_prob(pi).sum(-1, keepdim=True)
        actor_entropy = -log_pi.mean()

        # alpha step
        alpha_loss = self.log_alpha * (actor_entropy.detach() - self.target_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        lower_bound = self.critic(states, pi)[0].mean()
        actor_loss = -lower_bound + self.alpha * actor_entropy

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_critic_update()

        return {
            "bellman_surrogate": residual_error.item(),
            "pessimism_loss": pessimism_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_entropy": actor_entropy.item(),
            "lower_bound": lower_bound.item(),
            "action_diff": torch.mean(torch.norm(actions - pi, dim=1)).item(),
            "q_target": target_q.mean().item(),
        }
    
    @staticmethod
    def l2_projection(constraint: float):
         
         @torch.no_grad()
         def fn(module):
              if hasattr(module, "weight") and constraint > 0:
                   weight = module.weight
                   norm = torch.norm(weight)
                   weight.mul_(torch.clip(constraint / norm, max=1))
        
         return fn

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

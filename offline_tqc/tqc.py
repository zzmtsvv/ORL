from typing import Dict, Tuple
from copy import deepcopy
import torch
from config import tqc_config
from modules import Actor, TruncatedQuantileEnsembledCritic


class TQC:
    def __init__(self,
                 cfg: tqc_config,
                 actor: Actor,
                 critic: TruncatedQuantileEnsembledCritic) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.discount = cfg.discount
        self.batch_size = cfg.batch_size

        self.target_entropy = -float(actor.action_dim)

        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)

        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=cfg.alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = critic.to(self.device)
        self.critic_target = deepcopy(critic).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        self.quantiles_total = critic.num_critics * critic.num_quantiles
        self.quantiles2drop = cfg.quantiles_to_drop_per_critic * cfg.num_critics
        self.top = self.quantiles_total - self.quantiles2drop

        huber_tau = torch.arange(self.cfg.num_quantiles, device=self.device).float() / self.top + 1 / (2 * self.top)
        self.huber_tau = huber_tau[None, None, :, None]

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        # critic step
        critic_loss = self.critic_loss(states, actions, rewards, next_states, dones)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor step
        actor_loss, batch_entropy, qz_values = self.actor_loss(states)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha step
        alpha_loss = self.alpha_loss(states)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        self.soft_critic_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_batch_entropy": batch_entropy,
            "qz_values": qz_values,
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item()
        }
    
    def actor_loss(self, states: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        actions, log_prob = self.actor(states, need_log_prob=True)

        qz_values = self.critic(states, actions).mean(dim=2).mean(dim=1, keepdim=True)

        loss = self.alpha * log_prob - qz_values
        batch_entropy = -log_prob.mean().item()

        return loss.mean(), batch_entropy, qz_values.mean().item()

    def critic_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_states: torch.Tensor,
                    dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_states, need_log_prob=True)

            next_z = self.critic_target(next_states, next_actions)

            sorted_next_z = torch.sort(next_z.reshape(self.batch_size, -1)).values
            sorted_next_z_top = sorted_next_z[:, :self.top]
            sorted_next_z_top = sorted_next_z_top - self.alpha * next_log_prob.unsqueeze(-1)

            quantiles_target = rewards + self.discount * (1.0 - dones) * sorted_next_z_top
        
        current_z = self.critic(states, actions)
        loss = self.quantile_huber_loss(current_z, quantiles_target)

        return loss

    def quantile_huber_loss(self, quantiles: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pairwise_diff = target[:, None, None, :] - quantiles[:, :, :, None]
        abs_val = pairwise_diff.abs()
        
        huber_loss = torch.where(abs_val > 1.0,
                                 abs_val - 0.5,
                                 pairwise_diff.pow(2) / 2)

        loss = torch.abs(self.huber_tau - (pairwise_diff < 0).float()) * huber_loss
        return loss.mean()

    def alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, log_prob = self.actor(state, need_log_prob=True)
        
        loss = -self.log_alpha * (log_prob + self.target_entropy)
        return loss.mean()

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

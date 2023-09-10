from typing import Dict, Any, Tuple
import torch
from torch.nn import functional as F
from copy import deepcopy
from config import bear_config
from modules import Actor, EnsembledCritic


class BEAR:
    def __init__(self,
                 cfg: bear_config,
                 actor: Actor,
                 critic: EnsembledCritic,) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.discount = cfg.discount
        self.max_action = cfg.max_action

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = critic.to(self.device)
        self.critic_target = deepcopy(critic).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        self.mmd_kernel = self.gaussian_kernel if cfg.mmd_kernel_type == "gaussian" else self.laplacian_kernel
        self.mmd_sigma = cfg.mmd_sigma
        self.critic_lambda = cfg.critic_lambda

        self.target_entropy = -float(actor.action_dim)
        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=cfg.alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.log_lagrange = torch.randn((), requires_grad=True, device=self.device)
        self.lagrange_optim = torch.optim.AdamW([self.log_lagrange], lr=cfg.lagrange_lr)
        self.lagrange = self.log_lagrange.detach().exp()

        self.lagrange_threshold = cfg.lagrange_threshold

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1
        logging_dict = dict()

        # critic step
        critic_loss, q_mean = self.critic_loss(states,
                                       actions,
                                       rewards,
                                       next_states,
                                       dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        logging_dict["critic_loss"] = critic_loss.item()
        logging_dict["q_mean"] = q_mean

        # alpha step
        alpha_loss = self.alpha_loss(states)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # actor step
        actor_loss, sac_loss, mmd_loss, actor_batch_entropy, q_std = self.actor_loss(states, actions)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        logging_dict["actor_loss"] = actor_loss.mean().item()
        logging_dict["sac_loss"] = sac_loss
        logging_dict["actor_batch_entropy"] = actor_batch_entropy
        logging_dict["q_std"] = q_std
        logging_dict["mmd_loss"] = mmd_loss.mean().item()
        logging_dict["alpha"] = self.alpha.item()

        # lagrange step
        lagrange_loss = self.lagrange_loss(mmd_loss)

        self.lagrange_optim.zero_grad()
        lagrange_loss.backward()
        self.lagrange_optim.step()

        self.lagrange = self.log_lagrange.exp().detach()
        logging_dict["lagrange"] = self.lagrange.item()

        self.soft_critic_update()

        return logging_dict
    
    def lagrange_loss(self, mmd_loss: torch.Tensor) -> torch.Tensor:
        loss = self.log_lagrange.exp() * (mmd_loss - self.lagrange_threshold)

        return -loss.mean()
    
    def actor_loss(self,
                   states: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor, float, float]:
        pi, log_prob = self.actor(states, need_log_prob=True)
        q_values = self.critic(states, pi)

        q_value_min = q_values.min(0).values
        q_value_std = q_values.std(0)
        batch_entropy = -log_prob.mean().item()

        actor_loss = self.alpha * log_prob - q_value_min
        mmd_loss = self.mmd_loss(actions, pi)

        loss = actor_loss + self.lagrange * mmd_loss
        
        return loss.mean(), actor_loss.mean().item(), mmd_loss.detach(), batch_entropy, q_value_std.mean().item()
    
    def alpha_loss(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, log_prob = self.actor(states, need_log_prob=True)
        
        loss = -self.log_alpha * (log_prob + self.target_entropy)
        return loss.mean()
    
    def critic_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_states: torch.Tensor,
                    dones: torch.Tensor) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            next_actions, next_action_log_prob = self.actor(next_states, need_log_prob=True)
            
            tgt_q = self.critic_target(next_states, next_actions)
            tgt_q = self.critic_lambda * tgt_q.min(0).values + (1 - self.critic_lambda) * tgt_q.max(0).values
            tgt_q = tgt_q - self.alpha * next_action_log_prob
            
            q_target = rewards + self.discount * (1.0 - dones) * tgt_q.unsqueeze(-1)
        
        current_q = self.critic(states, actions)

        critic_loss = F.mse_loss(current_q, q_target.squeeze(-1))

        return critic_loss, current_q.mean().item()

    @staticmethod
    def gaussian_kernel(x: torch.Tensor,
                        y: torch.Tensor,
                        sigma: float) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add dimension for broadcasting
        y = y.unsqueeze(0)  # Add dimension for broadcasting
        diff = torch.norm(x - y, dim=2)
        kernel = torch.exp(-torch.pow(diff, 2) / (2 * sigma * sigma))

        return kernel
    
    @staticmethod
    def laplacian_kernel(x: torch.Tensor,
                         y: torch.Tensor,
                         sigma: float) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add dimension for broadcasting
        y = y.unsqueeze(0)  # Add dimension for broadcasting
        diff = torch.norm(x - y, dim=2)
        kernel = torch.exp(-diff / sigma)
        return kernel
    
    def mmd_loss(self,
                 x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        xx = self.mmd_kernel(x, x, self.mmd_sigma)
        xy = self.mmd_kernel(x, y, self.mmd_sigma)
        yy = self.mmd_kernel(y, y, self.mmd_sigma)

        loss = xx - 2 * xy + yy
        return loss

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optim": self.alpha_optimizer.state_dict(),
            "log_lagrange": self.log_lagrange,
            "lagrange_optim": self.lagrange_optim.state_dict(),
            "total_iterations": self.total_iterations
        }

    def load_state_dict(self, state_dict) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.log_alpha = state_dict["log_alpha"]
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_lagrange = state_dict["log_lagrange"]
        self.lagrange_optim.load_state_dict(state_dict["lagrange_optim"])
        self.total_iterations = state_dict["total_iterations"]


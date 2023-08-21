import torch
from torch.autograd import grad
from torch.nn import functional as F
from copy import deepcopy
from config import edac_config
from modules import Actor, EnsembledCritic
from typing import Dict, Any, Union, Tuple


_Number = Union[float, int]


class EDAC:
    def __init__(self,
                 cfg: edac_config,
                 actor: Actor,
                 actor_optim: torch.optim.Optimizer,
                 critic: EnsembledCritic,
                 critic_optim: torch.optim.Optimizer,
                 alpha_lr: float = 3e-4) -> None:
        
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.eta = cfg.eta
        self.gamma = cfg.gamma

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

        critic_loss = self.critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

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
                    done: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)
            q_next = self.critic_target(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)
        
        q_values = self.critic(state, action)
        critic_loss = F.mse_loss(q_values,  q_target.squeeze(1))
        diversity_loss = self.critic_diversity_loss(state, action)

        loss = critic_loss + self.eta * diversity_loss
        return loss
    
    def critic_diversity_loss(self,
                              state: torch.Tensor,
                              action: torch.Tensor) -> torch.Tensor:
        num_critics = self.critic.num_critics

        state = state.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        action = action.unsqueeze(0).repeat_interleave(num_critics, dim=0).requires_grad_(True)

        q_ensemble = self.critic(state, action)

        q_action_grad = grad(q_ensemble.sum(), action, retain_graph=True, create_graph=True)[0]
        q_action_grad = q_action_grad / (torch.norm(q_action_grad, p=2, dim=2).unsqueeze(-1) + 1e-10)
        q_action_grad = q_action_grad.transpose(0, 1)

        masks = torch.eye(num_critics, device=self.device).unsqueeze(0).repeat(q_action_grad.shape[0], 1, 1)

        q_action_grad = q_action_grad @ q_action_grad.permute(0, 2, 1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1, 2)).mean()
        grad_loss = grad_loss / (num_critics - 1)

        return grad_loss
    
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

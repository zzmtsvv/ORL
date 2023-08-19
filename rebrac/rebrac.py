from typing import Dict, Union, Any, Optional
from copy import deepcopy
import torch
from torch.nn import functional as F
from modules import DeterministicActor, EnsembledCritic
from config import rebrac_config


_Number = Union[float, int]


class ReBRAC:
    def __init__(self,
                 cfg: rebrac_config,
                 actor: DeterministicActor,
                 actor_optim: torch.optim.Optimizer,
                 critic: EnsembledCritic,
                 critic_optim: torch.optim.Optimizer) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.max_action = cfg.max_action
        self.critic_bc_coef = cfg.critic_bc_coef
        self.actor_bc_coef = cfg.actor_bc_coef
        self.policy_freq = cfg.policy_freq
        self.discount = cfg.discount
        
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_target = deepcopy(actor)

        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_target = deepcopy(critic)

        self.total_iterations = 0
        self.tau = cfg.tau
    
    def soft_actor_update(self):
        for param, tgt_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
    
    def train(self,
              state: torch.Tensor,
              action: torch.Tensor,
              reward: torch.Tensor,
              next_state: torch.Tensor,
              next_action: torch.Tensor) -> Dict[str, _Number]:
        logging_dict = dict()
        self.total_iterations += 1

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # print(next_state.shape, noise.shape)
            # print(self.actor_target(next_state).shape)
            next_policy = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            tgt_q = self.critic(next_state, next_action)
            bc_penalty = (next_action - next_policy).pow(2).sum(-1)

            tgt_q = tgt_q.min(0).values - self.critic_bc_coef * bc_penalty
            tgt_q = reward.squeeze(-1) + self.discount * tgt_q
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, tgt_q)

        logging_dict["critic_loss"] = critic_loss.item()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor update
        if not self.total_iterations % self.policy_freq:
            pi = self.actor(state)
            q = self.critic(state, pi)[0]
            denominator = q.abs().mean().detach()

            actor_loss = -q.mean() / denominator + self.actor_bc_coef * F.mse_loss(pi, action)
            logging_dict["actor_loss"] = actor_loss.item()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.soft_critic_update()
            self.soft_actor_update()
        
        return logging_dict
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "total_iterations": self.total_iterations
        }

    def load_state_dict(self,
                        state_dict: Dict[str, Any]):
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.total_iterations = state_dict["total_iterations"]

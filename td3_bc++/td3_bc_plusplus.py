from typing import Dict, Union, Any, Optional
from copy import deepcopy
import torch
from torch.nn import functional as F
from modules import DeterministicActor, EnsembledCritic
from config import td3bcplusplus_config


_Number = Union[float, int]


class TD3BC_PlusPlus:
    def __init__(self,
                 cfg: td3bcplusplus_config,
                 actor: DeterministicActor,
                 actor_optim: torch.optim.Optimizer,
                 critic: EnsembledCritic,
                 critic_optim: torch.optim.Optimizer) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.max_action = cfg.max_action
        self.alpha = cfg.alpha
        self.policy_freq = cfg.policy_freq
        self.discount = cfg.discount
        
        self.actor = actor
        self.actor_optim = actor_optim
        with torch.no_grad():
            self.actor_target = deepcopy(actor)

        self.critic = critic
        self.critic_optim = critic_optim
        with torch.no_grad():
            self.critic_target = deepcopy(critic)

        self.total_iterations = 0
        self.tau = cfg.tau

        self.lambda_gp = cfg.lambda_gp
        self.gradient_penalty_freq = cfg.gradient_penalty_freq
        self.critic_gradient_margin = cfg.critic_gradient_margin
    
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
              done: torch.Tensor) -> Dict[str, _Number]:
        logging_dict = dict()
        self.total_iterations += 1

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # print(next_state.shape, noise.shape)
            # print(self.actor_target(next_state).shape)
            next_policy = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            tgt_q = self.critic_target(next_state, next_policy)

            tgt_q = tgt_q.min(0).values
            tgt_q = reward + (1.0 - done) * self.discount * tgt_q.unsqueeze(-1)
        
        current_q = self.critic(state, action)
        qq = current_q.clone().detach()
        critic_loss = F.mse_loss(current_q, tgt_q.squeeze(1))

        if not self.total_iterations % self.gradient_penalty_freq:
            repeated_states = state.clone().detach().repeat(16, 1).requires_grad_(True)
            random_actions = torch.rand(
                size=(repeated_states.shape[0], self.actor.action_dim),
                requires_grad=True,
                device=self.device
            ) * 2.0 - self.max_action

            current_q_ = self.critic(repeated_states, random_actions)
            grad_q_wrt_random_action = torch.autograd.grad(
                outputs=current_q_.sum(),
                inputs=random_actions,
                create_graph=True
            )[0].norm(p=2, dim=-1)
            
            gradient_penalty = F.relu(grad_q_wrt_random_action - self.critic_gradient_margin).pow(2).mean()
            logging_dict["critic_gradient_penalty"] = gradient_penalty.item()
            
            critic_loss = critic_loss + gradient_penalty * self.lambda_gp

        logging_dict["critic_loss"] = critic_loss.item()

        self.critic_optim.zero_grad()
        critic_loss.backward()#retain_graph=True)
        self.critic_optim.step()

        # actor update
        if not self.total_iterations % self.policy_freq:
            pi = self.actor(state)
            q = self.critic(state, pi)[0]
            # denominator = q.abs().mean().detach()
            q_values = qq.detach().mean(dim=0).squeeze()
            lmbda = self.alpha / q_values.abs().mean()

            actor_loss = -lmbda * q.mean() + \
                (F.mse_loss(pi, action, reduction='none').mean(dim=-1) * qq).mean()

            # actor_loss = -q.mean() / denominator + self.actor_bc_coef * F.mse_loss(pi, action)
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

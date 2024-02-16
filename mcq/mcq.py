from typing import Dict
from copy import deepcopy
import torch
from config import mcq_config
from modules import Actor, EnsembledCritic, ConditionalVAE


class MCQ:
    def __init__(self,
                 cfg: mcq_config,
                 actor: Actor,
                 critic: EnsembledCritic,
                 vae: ConditionalVAE,
                 alpha_lr: float = 3e-4) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        
        self.target_entropy = -float(actor.action_dim)

        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)

        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.actor = actor.to(self.device)
        self.actor_target = deepcopy(actor).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = critic.to(self.device)
        self.critic_target = deepcopy(critic).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.vae = vae.to(self.device)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=cfg.vae_lr)

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        # train cvae
        vae_loss = self.vae.elbo_loss(states, actions)

        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()

        # get critic target value
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_states, need_log_prob=True)
            q_next = self.critic_target(next_states, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == dones.shape == rewards.shape
            q_target = rewards + self.gamma * (1 - dones) * q_next.unsqueeze(-1)
        
        # sample actions from pi based on each states and next_states
        pass

        # compute the target value for the OOD actions
        pass

        # update critic
        pass
    
        # update actor
        pass
    
        self.soft_critic_update()

        return {
            "vae_loss": vae_loss.item(),
        }

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

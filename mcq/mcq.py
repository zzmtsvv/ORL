from typing import Dict
from copy import deepcopy
import torch
from torch.nn import functional as F
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
        self.num_samples = cfg.num_samples
        self.lmbda = cfg.lmbda
        
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
        vae_loss = self.vae.importance_sampling_loss(states, actions, num_samples=self.num_samples).mean()

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
        states_in = torch.cat([states, next_states], dim=0).repeat(self.num_samples, 1, 1).reshape(-1, self.cfg.state_dim)

        actions_ood, _ = self.actor(states_in)
        # [num_samples, 2 * batch_size, action_dim]
        # actions_ood = actions_ood.repeat(self.num_samples, 1, 1).reshape(-1, self.cfg.action_dim)

        # compute the target value for the OOD actions
        actions_in = self.vae.decode(states_in).reshape(-1, self.cfg.action_dim)
        q_values_in = self.critic(states_in, actions_in).reshape(self.cfg.num_critics, self.num_samples, -1)
        pseudo_target = torch.max(q_values_in, dim=1)[0].min(0).values  # [2 * batch_size,]
        # idk why bit this weird stuff in the original implementation
        pseudo_target = pseudo_target.clamp_min(0).reshape(-1, 1)  # [2 * batch_size * num_samples, 1]

        q_values_ood = self.critic(states_in, actions_ood).unsqueeze(-1)  # [num_critics, 2 * num_samples * batch_size, 1]

        # update critic
        q_values = self.critic(states, actions).unsqueeze(-1)
        critic_main_loss = F.mse_loss(q_values, q_target)
        critic_ood_loss = F.mse_loss(q_values_ood, pseudo_target.repeat(self.num_samples, 1))
        critic_loss = self.lmbda * critic_main_loss + (1. - self.lmbda) * critic_ood_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
    
        # update actor
        pi, log_prob = self.actor(states, need_log_prob=True)
        q_values = self.critic(states, pi)

        q_value_min = q_values.min(0).values
        batch_entropy = -log_prob.mean().item()

        assert log_prob.shape == q_value_min.shape
        actor_loss = (self.alpha * log_prob - q_value_min).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update alpha
        with torch.no_grad():
            action, log_prob = self.actor(states, need_log_prob=True)
        
        alpha_loss = (-self.log_alpha * (log_prob + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()
    
        self.soft_critic_update()

        return {
            "vae_loss": vae_loss.item(),
            "bellman_critic_loss": critic_main_loss.item(),
            "critic_ood_loss": critic_ood_loss.item(),
            "batch_actor_entropy": batch_entropy,
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item()
        }

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

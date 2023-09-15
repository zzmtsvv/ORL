from math import pow
from typing import Tuple, Dict, Any
import torch
from torch.nn import functional as F
from copy import deepcopy
from config import td7_config
from modules import TD7Actor, TD7Critic, TD7Encoder


class TD7:
    def __init__(self,
                 cfg: td7_config,
                 encoder: TD7Encoder,
                 actor: TD7Actor,
                 critic: TD7Critic,) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.discount = cfg.discount
        self.expl_noise = cfg.exploration_noise
        self.policy_noise = cfg.policy_noise
        self.policy_freq = cfg.policy_freq
        self.noise_clip = cfg.noise_clip
        self.target_update_freq = cfg.target_update_freq

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.actor_target = deepcopy(actor).to(self.device)

        self.critic = critic.to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        self.critic_target = deepcopy(critic).to(self.device)

        self.encoder = encoder.to(self.device)
        self.encoder_optim = torch.optim.AdamW(self.encoder.parameters(), lr=cfg.encoder_lr)
        self.fixed_encoder = deepcopy(encoder).to(self.device)
        self.fixed_encoder_target = deepcopy(encoder).to(self.device)

        self.min_priority = cfg.min_priority
        self.max_action = cfg.max_action

        self.max_target = 0
        self.min_target = 0

        self.running_max_q = -float("inf")
        self.running_min_q = float("inf")

        self.lambda_coef = cfg.lambda_coef
        self.alpha = cfg.alpha

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Tuple[Dict[str, float], torch.Tensor]:
        self.total_iterations += 1
        logging_dict = dict()

        # encoder step
        encoder_loss = self.encoder_loss(states, actions, next_states)

        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        self.encoder_optim.step()

        logging_dict["encoder_loss"] = encoder_loss.item()

        # critic step
        critic_loss, q_values, priority, q_max, q_min = self.critic_loss(states,
                                                                         actions,
                                                                         rewards,
                                                                         next_states,
                                                                         dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        logging_dict["critic_loss"] = critic_loss.item()
        logging_dict["q_values"] = q_values.item()
        logging_dict["q_max"] = q_max
        logging_dict["q_min"] = q_min

        # actor step
        if not self.total_iterations % self.policy_freq:
            actor_loss, bc_loss = self.actor_loss(states, actions)

            self.actor_optim.zero_grad()
            (actor_loss + self.lambda_coef * bc_loss).backward()
            self.actor_optim.step()

            logging_dict["actor_loss"] = actor_loss.item()
            logging_dict["bc_loss"] = bc_loss.item()

        if not self.total_iterations % self.target_update_freq:
            self.update_target_models()

        return logging_dict, priority
    
    def actor_loss(self,
                   states: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fixed_zs = self.fixed_encoder.f(states)
        new_actions = self.actor(states, fixed_zs)
        fixed_zsa = self.fixed_encoder.g(fixed_zs, new_actions)
        q = self.critic(states, new_actions, fixed_zsa, fixed_zs)

        actor_loss = -q.mean()
        bc_loss = q.abs().mean().detach() * F.mse_loss(new_actions, actions)

        return actor_loss, bc_loss
    
    def encoder_loss(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_zs = self.encoder.f(next_states)
        zs = self.encoder.f(states)
        current_zsa = self.encoder.g(zs, actions)

        encoder_loss = F.mse_loss(current_zsa, next_zs)
        return encoder_loss

    def critic_loss(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        with torch.no_grad():
            fixed_target_next_zs = self.fixed_encoder_target.f(next_states)
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states, fixed_target_next_zs) + noise).clamp(-self.max_action, self.max_action)

            fixed_target_next_zsa = self.fixed_encoder_target.g(fixed_target_next_zs, next_actions)
            target_q = self.critic_target(next_states, next_actions, fixed_target_next_zsa, fixed_target_next_zs).min(0).values

            target_q = rewards + self.discount * (1.0 - dones) * target_q.clamp(self.min_target, self.max_target)

            self.running_max_q = max(self.running_max_q, target_q.max().item())
            self.running_min_q = min(self.running_min_q, target_q.min().item())

            fixed_zs = self.fixed_encoder.f(states)
            fixed_zsa = self.fixed_encoder.g(fixed_zs, actions)
        
        current_q = self.critic(states, actions, fixed_zsa, fixed_zs)
        td = (target_q - current_q).abs()
        critic_loss = self.huber_loss(td)

        priority = td.max(0).values.detach().squeeze(-1).clamp_min(self.min_priority).pow(self.alpha)

        return critic_loss, current_q.mean(), priority, self.running_max_q, self.running_min_q

    def huber_loss(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < self.min_priority, x.pow(2) / 2, x).sum(dim=0).mean()
    
    def pal(self, td: torch.Tensor) -> torch.Tensor:
        '''
            Prioritized Approximation Loss
            is used with uniform sampling from the replay buffer
        '''
        if self.min_priority == 1.0:
            loss = torch.where(
                td.abs() < 1.0,
                td.pow(2) / 2,
                td.abs().pow(1.0 + self.alpha) / (1.0 + self.alpha)
            ).mean()
        else:
            loss = torch.where(
                td.abs() < self.min_priority,
                pow(self.min_priority, self.alpha) * td.pow(2) / 2,
                self.min_priority * td.abs().pow(1.0 + self.alpha) / (1.0 + self.alpha)
            ).mean()
        
        lambda_coef = td.abs().clamp_min(self.min_priority).pow(self.alpha).mean().detach()
        return loss / lambda_coef

    def update_target_models(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
        self.fixed_encoder.load_state_dict(self.encoder.state_dict())

        self.max_target = self.running_max_q
        self.min_target = self.running_min_q

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "actor": self.actor.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "encoder": self.encoder.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict(),
            "fixed_encoder": self.fixed_encoder.state_dict(),
            "fixed_encoder_target": self.fixed_encoder_target.state_dict(),
            "max_target": self.max_target,
            "min_target": self.min_target,
            "running_max_q": self.running_max_q,
            "running_min_q": self.running_min_q
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.total_iterations = state_dict["total_iterations"]
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.encoder.load_state_dict(state_dict["encoder"])
        self.encoder_optim.load_state_dict(state_dict["encoder_optim"])
        self.fixed_encoder.load_state_dict(state_dict["fixed_encoder"])
        self.fixed_encoder_target.load_state_dict(state_dict["fixed_encoder_target"])
        self.max_target = state_dict["max_target"]
        self.min_target = state_dict["min_target"]
        self.running_max_q = state_dict["running_max_q"]
        self.running_min_q = state_dict["running_min_q"]

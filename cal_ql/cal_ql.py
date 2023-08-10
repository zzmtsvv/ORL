import numpy as np
from typing import Tuple, Dict, Any, Union
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from config import calql_config
from modules import EnsembledCritic, TanhGaussianPolicy


_Number = Union[float, int]


class CalQL:
    def __init__(self,
                 cfg: calql_config) -> None:
        self.cfg = cfg

        self.max_action = cfg.max_action
        self.device = cfg.device
        self.tau = cfg.tau
        self.target_entropy = cfg.target_entropy
        self.alpha_multiplier = cfg.alpha_multiplier
        self.bc_steps = cfg.bc_steps
        self.use_max_target_backup = cfg.use_max_target_backup
        self.num_actions = cfg.num_actions
        self.backup_entropy = cfg.backup_entropy
        self.discount_factor = cfg.discount_factor

        self.critic = EnsembledCritic(cfg.state_dim,
                                      cfg.action_dim,
                                      cfg.hidden_dim,
                                      orthogonal_init=cfg.orthogonal_init).to(self.device)
        self.target_critic = deepcopy(self.critic)

        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        self.actor = TanhGaussianPolicy(cfg.state_dim,
                                        cfg.action_dim,
                                        cfg.max_action,
                                        orthogonal_initialization=cfg.orthogonal_init,
                                        hidden_dim=cfg.hidden_dim).to(self.device)
        
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.log_alpha = None
        self.use_automatic_entropy_tuning = cfg.use_automatic_entropy_tuning
        if cfg.use_automatic_entropy_tuning:
            self.log_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=cfg.device, requires_grad=True))
            self.alpha_optim = torch.optim.AdamW(
                self.log_alpha, lr=cfg.actor_lr
            )

        self.log_alpha_prime = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=cfg.device, requires_grad=True))
        self.alpha_prime_optim = torch.optim.AdamW(self.log_alpha_prime, lr=cfg.critic_lr)

        self.calibration = True
        self.total_iterations = 0

        self.temperature = self.cfg.temperature
        self.random_density = self.cfg.action_dim * np.log(0.5)  # used for importance sampling
        self.num_critics = self.critic.num_critics
    
    def target_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def switch_calibration(self):
        self.calibration = not self.calibration
    
    def alpha_loss(self,
                   states: torch.Tensor,
                   log_policy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            returns alpha and alpha loss
        '''
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_policy + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp() * self.alpha_multiplier
        else:
            alpha_loss = states.new_tensor(0.0)
            alpha = states.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss
    
    def actor_loss(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   new_actions: torch.Tensor,
                   alpha: torch.Tensor,
                   log_policy: torch.Tensor) -> torch.Tensor:
        if self.total_iterations < self.bc_steps:
            log_probs = self.actor.log_prob(states, actions)

            actor_loss = (alpha * log_policy - log_probs).mean()
        else:
            q_new = self.critic(states, new_actions).min(0).values

            actor_loss = (alpha * log_policy - q_new).mean()

        return actor_loss
    
    def critic_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    next_states: torch.Tensor,
                    rewards: torch.Tensor,
                    dones: torch.Tensor,
                    mc_returns: torch.Tensor,
                    alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, _Number]]:
        q_predicted = self.critic(states, actions)

        with torch.no_grad():
            if self.use_max_target_backup:
                new_next_actions, next_log_policy = self.actor(
                    next_states, num_repeats=self.num_actions
                )
                target_q, max_target_indexes = torch.max(
                    self.target_critic(next_states, new_next_actions).min(-1)
                )
                next_log_policy = torch.gather(
                    next_log_policy, -1, max_target_indexes.unsqueeze(-1)
                ).squeeze(-1)
            else:
                new_next_actions, next_log_policy = self.actor(next_states)
                target_q = self.target_critic(next_states, new_next_actions).min(0)
            
            if self.backup_entropy:
                target_q = target_q - alpha * next_log_policy
        
        target_q = target_q.unsqueeze(-1)
        target_q = rewards + (1.0 - dones) * self.discount_factor * target_q.detach()
        target_q = target_q.squeeze(-1).detach()

        critic_loss = F.mse_loss(q_predicted, target_q)

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        random_actions = actions.new_empty(
            (batch_size, self.num_actions, action_dim), requires_grad=False
        ).uniform_(-self.max_action, self.max_action)

        current_actions, current_log_pi = self.actor(
            states, num_repeats=self.num_actions
        )
        next_actions, next_log_pi = self.actor(
            next_states, num_repeats=self.num_actions
        )
        current_actions = current_actions.detach()
        current_log_pi = current_log_pi.detach()
        next_actions = next_actions.detach()
        next_log_pi = next_log_pi.detach()

        # [2, batch_size, num_actions]
        q_random = self.critic(states, random_actions)
        q_current_actions = self.critic(states, current_actions)
        q_next_actions = self.critic(states, next_actions)

        # calibration
        lower_bounds = mc_returns.reshape(-1, 1).repeat(1, q_current_actions.shape[-1])
        num_values = lower_bounds.numel()

        bound_rate_q_current_actions = torch.sum(q_current_actions < lower_bounds, dim=-1) / num_values
        bound_rate_q_next_actions = torch.sum(q_next_actions < lower_bounds, dim=-1) / num_values

        if self.calibration:
            q_current_actions = torch.maximum(q_current_actions, lower_bounds)
            q_next_actions = torch.maximum(q_next_actions, lower_bounds)
        
        # concat along num_actions dim
        cql_cat_q = torch.cat([
            q_random,
            q_predicted.unsqueeze(-1),
            q_next_actions,
            q_current_actions
        ], dim=-1)

        cql_std_q = torch.std(cql_cat_q, dim=-1)

        if self.cfg.importance_sampling:
            
            cql_cat_q = torch.cat([
                q_random - self.random_density,
                q_next_actions - next_log_pi.detach(),
                q_current_actions - current_log_pi.detach()
            ], dim=-1)
        
        cql_q_ood = torch.logsumexp(cql_cat_q / self.temperature, dim=-1) * self.temperature

        cql_q_diff = torch.clamp(
            cql_q_ood - q_predicted,
            self.cfg.clip_diff_min,
            self.cfg.clip_diff_max).mean(dim=-1)
        
        if self.cfg.use_lagrange:
            alpha_prime = torch.exp(self.log_alpha_prime).clamp(min=0.0, max=1000000.0)

            cql_min_q_loss = alpha_prime * self.cfg.alpha * (cql_q_diff - self.cfg.target_action_gap)
            self.alpha_prime_optim.zero_grad()
            alpha_prime_loss = -cql_min_q_loss.sum() / self.num_critics
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optim.step()

        else:
            cql_min_q_loss = self.cfg.alpha * cql_q_diff
            alpha_prime_loss = states.new_tensor(0.0)
            alpha_prime = states.new_tensor(0.0)
        
        final_critic_loss = critic_loss.sum() + cql_min_q_loss.sum()

        logging_dict = {
            "offline_cql/std_q": cql_std_q.mean().item(),
            "cql/q_random": q_random.mean().item(),
            "cql/min_q_loss": cql_min_q_loss.mean().item(),
            "cql/q_diff": cql_q_diff.mean().item(),
            "cql/q_current_actions": q_current_actions.mean().item(),
            "cql/q_next_actions": q_next_actions.mean().item(),
            "alpha_prime_loss": alpha_prime_loss.item(),
            "alpha_prime": alpha_prime.item(),
            "bound_rate/q_current_actions": bound_rate_q_current_actions.mean().item(),
            "bound_rate/q_next_actions": bound_rate_q_next_actions.mean().item()
        }

        return final_critic_loss, alpha_prime, alpha_prime_loss, logging_dict

    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor,
              mc_returns: torch.Tensor) -> Dict[str, _Number]:
        self.total_iterations += 1

        new_action, log_pi = self.actor(states)
        alpha, alpha_loss = self.alpha_loss(states, log_pi)

        policy_loss = self.actor_loss(states, actions, new_action, alpha, log_pi)

        logging_dict = {
            "log_policy": log_pi.mean().item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item()
        }

        critic_loss, alpha_prime, alpha_prime_loss, log_dict = self.critic_loss(states,
                                                                                actions,
                                                                                next_states,
                                                                                rewards,
                                                                                dones,
                                                                                mc_returns,
                                                                                alpha)
        logging_dict.update(log_dict)

        if self.use_automatic_entropy_tuning:
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        if not self.total_iterations % self.cfg.target_update_period:
            self.target_critic_update()
        
        return logging_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.target_critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optim.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optim.state_dict(),
            "total_iterations": self.total_iterations,
        }

    def load_state_dict(self,
                        state_dict: Dict[str, Any]):
        
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic.load_state_dict(state_dict=state_dict["critic"])

        self.target_critic.load_state_dict(state_dict=state_dict["critic_target"])

        self.critic_optim.load_state_dict(
            state_dict=state_dict["critic_optim"]
        )
        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optim.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optim.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_iterations = state_dict["total_iterations"]
        self.actor_optim.load_state_dict(state_dict=state_dict["actor_optim"])

from typing import Tuple, Dict, Any
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from config import calql_config
from modules import FullyConnectedCritic, TanhGaussianPolicy


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

        self.critic1 = FullyConnectedCritic(cfg.state_dim,
                                            cfg.action_dim,
                                            cfg.orthogonal_init,
                                            cfg.hidden_dim).to(self.device)
        self.critic2 = FullyConnectedCritic(cfg.state_dim,
                                            cfg.action_dim,
                                            cfg.orthogonal_init,
                                            cfg.hidden_dim).to(self.device)
        
        self.target_critic1 = deepcopy(self.critic1).to(self.device)
        self.target_critic2 = deepcopy(self.critic2).to(self.device)

        self.critic1_optim = torch.optim.AdamW(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_optim = torch.optim.AdamW(self.critic2.parameters(), lr=cfg.critic_lr)

        self.actor = TanhGaussianPolicy(cfg.state_dim,
                                        cfg.action_dim,
                                        cfg.max_action,
                                        orthogonal_initialization=cfg.orthogonal_init,
                                        hidden_dim=cfg.hidden_dim)
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
    
    def target_update(self):
        for param, tgt_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

        for param, tgt_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def switch_calibration(self):
        self.calibration = not self.calibration
    
    def alpha_loss(self,
                   states: torch.Tensor,
                   log_policy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            q_new = torch.min(
                self.critic1(states, new_actions),
                self.critic2(states, new_actions)
            )
            actor_loss = (alpha * log_policy - q_new).mean()
        return actor_loss
    
    def critic_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    next_states: torch.Tensor,
                    rewards: torch.Tensor,
                    dones: torch.Tensor,
                    mc_returns: torch.Tensor,
                    alpha: torch.Tensor):
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        if self.use_max_target_backup:
            new_next_actions, next_log_policy = self.actor(
                next_states, num_repeats=self.num_actions
            )
            target_q, max_target_indexes = torch.max(
                torch.min(
                    self.target_critic1(next_states, new_next_actions),
                    self.target_critic2(next_states, new_next_actions),
                ),
                dim=-1,
            )
            next_log_policy = torch.gather(
                next_log_policy, -1, max_target_indexes.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_policy = self.actor(next_states)
            target_q = torch.min(
                self.target_critic1(next_states, new_next_actions),
                self.target_critic2(next_states, new_next_actions),
            )
        
        if self.backup_entropy:
            target_q = target_q - alpha * next_log_policy
        
        target_q = target_q.unsqueeze(-1)
        target_q = rewards + (1.0 - dones) * self.discount_factor * target_q.detach()
        target_q = target_q.squeeze(-1).detach()

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

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
        current_actions, current_log_pi = (
            current_actions.detach(),
            current_log_pi.detach(),
        )
        next_actions, next_log_pi = (
            next_actions.detach(),
            next_log_pi.detach(),
        )

        q1_random = self.critic1(states, random_actions)
        q2_random = self.critic2(states, random_actions)
        q1_current_actions = self.critic1(states, current_actions)
        q2_current_actions = self.critic2(states, current_actions)
        q1_next_actions = self.critic1(states, next_actions)
        q2_next_actions = self.critic2(states, next_actions)

        raise NotImplementedError("Q Loss is not ready yet")

    def train(self,):
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.target_critic1.state_dict(),
            "critic2_target": self.target_critic2.state_dict(),
            "critic1_optim": self.critic1_optim.state_dict(),
            "critic2_optim": self.critic2_optim.state_dict(),
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
        self.critic1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic1_optim.load_state_dict(
            state_dict=state_dict["critic1_optim"]
        )
        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optim.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optim.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_iterations"]
        self.critic2_optim.load_state_dict(
            state_dict=state_dict["critic2_optimizer"]
        )
        self.actor_optim.load_state_dict(state_dict=state_dict["actor_optim"])


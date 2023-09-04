from typing import Dict, Union, Any
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from modules import DeterministicActor, EnsembledCritic
from config import prdc_config

from scipy.spatial import KDTree


_Number = Union[float, int]


class PRDC:
    def __init__(self,
                 cfg: prdc_config,
                 kdtree_data: np.ndarray,
                 actor: DeterministicActor,
                 critic: EnsembledCritic) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.max_action = cfg.max_action
        self.discount = cfg.discount
        self.policy_freq = cfg.policy_freq

        self.alpha = cfg.alpha
        self.k = cfg.k
        self.beta = cfg.beta

        self.actor = actor.to(self.device)
        self.actor_target = deepcopy(actor).to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)


        self.critic = critic.to(self.device)
        self.critic_target = deepcopy(critic).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        self.kdtree_data = kdtree_data
        self.kd_tree = KDTree(kdtree_data)
        
        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, _Number]:
        logging_dict = dict()
        self.total_iterations += 1

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            tgt_q = self.critic_target(next_states, next_actions).min(0).values
            
            q_target = rewards + self.discount * (1.0 - dones) * tgt_q.unsqueeze(-1)
        
        # critic step
        current_q = self.critic(states, actions)

        critic_loss = F.mse_loss(current_q, q_target.squeeze(-1))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        logging_dict["critic_loss"] = critic_loss.item()

        # actor step
        if not self.total_iterations % self.policy_freq:
            pi = self.actor(states)
            q = self.critic(states, pi)[0]
            denominator = q.abs().mean().detach()
            lambda_ = self.alpha / denominator

            actor_loss = -lambda_ * q.mean()

            key = torch.cat((self.beta * states, pi), dim=-1).detach().cpu().numpy()
            _, index = self.kd_tree.query(key, k=self.k, workers=-1)
            knn = torch.tensor(self.kdtree_data[index][:, -self.cfg.action_dim:]).squeeze(1).to(self.device)

            dc_loss = F.mse_loss(pi, knn)

            overall_actor_loss = actor_loss + dc_loss

            self.actor_optim.zero_grad()
            overall_actor_loss.backward()
            self.actor_optim.step()

            logging_dict["q_value"] = q.mean().item()
            logging_dict["actor_lambda"] = lambda_.item()
            logging_dict["actor_loss"] = overall_actor_loss.item()
            logging_dict["dc_loss"] = dc_loss.item()
            logging_dict["actor_td3_loss"] = actor_loss.item()

            self.soft_critic_update()
            self.soft_actor_update()

        return logging_dict

    def soft_actor_update(self):
        for param, tgt_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "total_iterations": self.total_iterations
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.total_iterations = state_dict["total_iterations"]

import torch
from torch.nn import functional as F
from config import crossq_config
from modules import Actor, Critic
from typing import Dict, Any, Union, Tuple


_Number = Union[float, int]


class CrossQ:
    def __init__(self,
                 cfg: crossq_config,
                 actor: Actor,
                 critic1: Critic,
                 critic2: Critic,
                 alpha_lr: float = 3e-4) -> None:
        
        self.cfg = cfg
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.policy_delay = cfg.actor_delay

        self.target_entropy = -float(actor.action_dim)

        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)

        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=(0.5, 0.999))

        self.critic1 = critic1.to(self.device)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr, betas=(0.5, 0.999))

        self.critic2 = critic2.to(self.device)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr, betas=(0.5, 0.999))

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, _Number]:
        self.total_iterations += 1

        logging_dict = dict()

        alpha_loss = self.alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        logging_dict["alpha_loss"] = alpha_loss.item()
        logging_dict["alpha"] = self.alpha.item()

        if self.total_iterations % self.policy_delay == 0:
            actor_loss, batch_entropy, q_policy_std = self.actor_loss(states)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            logging_dict["actor_loss"] = actor_loss.item()
            logging_dict["batch_entropy"] = batch_entropy

        critic_loss1, critic2_loss = self.critic_loss(states, actions, rewards, next_states, dones)
        
        self.critic1_optim.zero_grad()
        critic_loss1.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        
        logging_dict["critic_loss"] = (critic_loss1.item() + critic2_loss.item()) / 2
        
        return logging_dict

    def alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, log_prob = self.actor(state, need_log_prob=True)
        
        loss = -self.log_alpha * (log_prob + self.target_entropy)
        return loss.mean()
    
    def actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, log_prob = self.actor(state, need_log_prob=True)
        q_values1 = self.critic1(state, action)
        q_values2 = self.critic2(state, action)

        q_value_min = torch.min(q_values1, q_values2)
        q_value_std = torch.cat([q_values1, q_values2]).std().item()
        batch_entropy = -log_prob.mean().item()

        # assert log_prob.shape == q_value_min.shape
        loss = self.alpha * log_prob - q_value_min

        return loss.mean(), batch_entropy, q_value_std

    def critic_loss(self,
                    state: torch.Tensor,
                    action: torch.Tensor,
                    reward: torch.Tensor,
                    next_state: torch.Tensor,
                    done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)
            # q_next = self.critic_target(next_state, next_action).min(0).values
            # q_next = q_next - self.alpha * next_action_log_prob

            # assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            # q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)
        all_q1 = self.critic1(torch.cat([state, next_state], dim=0),
                              torch.cat([action, next_action], dim=0))
        all_q2 = self.critic2(torch.cat([state, next_state], dim=0),
                              torch.cat([action, next_action], dim=0))
        q1, next_q1 = all_q1.chunk(2, dim=0)
        q2, next_q2 = all_q2.chunk(2, dim=0)

        q_next = torch.min(next_q1, next_q2).detach()
        q_next = q_next - self.alpha * next_action_log_prob.unsqueeze(-1)

        assert q_next.shape == done.shape == reward.shape
        q_target = reward + self.gamma * (1 - done) * q_next

        # q_values = torch.cat([q1, q2], dim=-1)
        # critic_loss = F.mse_loss(q_values,  q_target.squeeze(1))
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        return critic1_loss, critic2_loss

    # def state_dict(self) -> Dict[str, Any]:
    #     return {
    #         "actor": self.actor.state_dict(),
    #         "critic": self.critic.state_dict(),
    #         "target_critic": self.critic_target.state_dict(),
    #         "log_alpha": self.log_alpha.item(),
    #         "actor_optimizer": self.actor_optim.state_dict(),
    #         "critic_optimizer": self.critic_optim.state_dict(),
    #         "alpha_optimizer": self.alpha_optimizer.state_dict(),
    #     }

    # def load_state_dict(self, state_dict: Dict[str, Any]):
    #     self.actor.load_state_dict(state_dict["actor"])
    #     self.critic.load_state_dict(state_dict["critic"])
    #     self.critic_target.load_state_dict(state_dict["target_critic"])
    #     self.actor_optim.load_state_dict(state_dict["actor_optimizer"])
    #     self.critic_optim.load_state_dict(state_dict["critic_optimizer"])
    #     self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
    #     self.log_alpha.data[0] = state_dict["log_alpha"]
    #     self.alpha = self.log_alpha.exp().detach()

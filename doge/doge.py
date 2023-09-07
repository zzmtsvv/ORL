from typing import Dict, Any, Tuple
import torch
from torch.nn import functional as F
from copy import deepcopy
from config import doge_config
from modules import DeterministicActor, EnsembledCritic, Distance


class DOGE:
    def __init__(self,
                 cfg: doge_config,
                 actor: DeterministicActor,
                 critic: EnsembledCritic,
                 distance_fn: Distance) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.discount = cfg.discount
        self.policy_noise = cfg.policy_noise
        self.policy_freq = cfg.policy_freq
        self.max_action = cfg.max_action
        self.noise_clip = cfg.noise_clip
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.distance_steps = cfg.distance_steps

        self.actor = actor.to(self.device)
        with torch.no_grad():
            self.actor_target = deepcopy(actor).to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = critic.to(self.device)
        with torch.no_grad():
            self.critic_target = deepcopy(critic).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        self.distance = distance_fn.to(self.device)
        self.distance_optim = torch.optim.AdamW(self.distance.parameters(), lr=cfg.distance_lr)

        self.lambda_prime = torch.tensor(cfg.initial_lambda)
        self.dual_step_size = torch.tensor(cfg.lambda_lr)
        
        self.total_iterations = 0

    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1
        logging_dict = dict()

        # critic step
        critic_loss = self.critic_loss(states,
                                       actions,
                                       rewards,
                                       next_states,
                                       dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        logging_dict["critic_loss"] = critic_loss.item()

        # distance step
        distance_loss = 0.0
        if self.total_iterations <= self.distance_steps:
            distance_loss = self.distance_loss(states, actions)

            self.distance_optim.zero_grad()
            distance_loss.backward()
            self.distance_optim.step()

        logging_dict["distance_loss"] = distance_loss.item()

        # actor step
        if not self.total_iterations % self.policy_freq:
            actor_loss, bc_loss, q_mean, distance_diff, distance_mean = self.actor_loss(states, actions)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            logging_dict["actor_loss"] = actor_loss.item()
            logging_dict["bc_loss"] = bc_loss.item()
            logging_dict["q_value"] = q_mean.item()
            logging_dict["distance_difference"] = distance_diff.item(),
            logging_dict["distance_mean"] = distance_mean
            logging_dict["dual_grad_descent_lambda"] = self.lambda_prime.item()

            self.soft_critic_update()
            self.soft_actor_update()

        return logging_dict

    def critic_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_states: torch.Tensor,
                    dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            tgt_q = self.critic_target(next_states, next_actions).min(0).values
            
            q_target = rewards + self.discount * (1.0 - dones) * tgt_q.unsqueeze(-1)
        
        current_q = self.critic(states, actions)

        critic_loss = F.mse_loss(current_q, q_target.squeeze(-1))

        return critic_loss

    def distance_loss(self,
                      states: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
        prediction, label = self.distance.linear_distance(states, actions)
        return F.mse_loss(prediction, label)

    def actor_loss(self,
                   states: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = self.actor(states)
        q_values = self.critic(states, pi)[0]

        lambda_ = self.alpha / q_values.abs().mean().detach()

        bc_loss = F.mse_loss(pi, actions)

        distance = self.distance.value(states, pi)
        distance_diff = (distance - self.distance.value(states, actions).detach()).mean()
        
        self.lambda_prime_update(distance_diff)
        distance_penalty = self.lambda_prime.detach() * (distance_diff - self.cfg.lambda_threshold)
        distance_mean = distance.mean()

        actor_loss = -lambda_ * q_values.mean() + distance_penalty

        print(distance_diff)
        return

        return actor_loss, bc_loss, q_values.mean(), distance_diff, distance_mean

    def lambda_prime_update(self, distance_difference: torch.Tensor) -> None:
        with torch.no_grad():
            base_loss = distance_difference - self.cfg.lambda_threshold
            lambda_loss = self.lambda_prime * base_loss

            self.lambda_prime += self.dual_step_size * lambda_loss.cpu().item()
            self.lambda_prime.clip_(self.cfg.lambda_min, self.cfg.lambda_max)

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
            "distance_fn": self.distance.state_dict(),
            "distance_optim": self.distance_optim.state_dict(),
            "total_iterations": self.total_iterations
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.distance.load_state_dict(state_dict["distance_fn"])
        self.distance_optim.load_state_dict(state_dict["distance_optim"])
        self.total_iterations = state_dict["total_iterations"]
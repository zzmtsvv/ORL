from copy import deepcopy

import torch
from torch.nn import functional as F

from critic_modules import EnsembledCritic
from actor_modules import ActorVectorField, OneStepPolicy


class FQL:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 bc_alpha: float = 0.2,
                 num_flow_steps: int = 10,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 num_critics: int = 2,
                 min_action: float = -1.0,
                 max_action: float = 1.0,
                 device: str = "cuda",
                 ) -> None:
        self.device = device
        
        self.num_critics = num_critics
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.bc_alpha = bc_alpha
        self.num_flow_steps = num_flow_steps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount = discount
        self.tau = tau
        self.min_action = min_action
        self.max_action = max_action

        self.actor_bc_flow = ActorVectorField(state_dim, action_dim, hidden_dim).to(device)
        self.actor_bc_flow_optimizer = torch.optim.Adam(self.actor_bc_flow.parameters(), lr=actor_lr)

        self.one_step_actor = OneStepPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.one_step_actor_optimizer = torch.optim.Adam(self.one_step_actor.parameters(), lr=actor_lr)
        
        self.critic = EnsembledCritic(state_dim, action_dim, hidden_dim, num_critics).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)
        
        self.total_iterations = 0
    
    def train_step(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
            rng: torch.Generator | None = None
    ) -> dict[str, float | int]:
        batch_size = states.shape[0]
        # critic loss
        with torch.no_grad():
            next_actions = self.sample_actions(next_states, rng)
            
            q_next = self.target_critic(next_states, next_actions).mean(0)

            assert q_next.unsqueeze(-1).shape == dones.shape == rewards.shape
            q_target = rewards + self.discount * (1.0 - dones) * q_next.unsqueeze(-1)
        
        q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(q_values,  q_target.squeeze(1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # bc flow loss
        x_0 = torch.randn(
            size=(batch_size, self.action_dim),
            device=self.device,
            generator=rng
        )
        # x_1 is cosidered to be actions
        timesteps = torch.rand(
            size=(batch_size, 1),
            device=self.device,
            generator=rng
        )
        x_t = (1.0 - timesteps) * x_0 + timesteps * actions
        velocities = actions - x_0

        predictions = self.actor_bc_flow(states, x_t, timesteps)
        bc_flow_loss = F.mse_loss(predictions, velocities)

        # distillation loss
        noises = torch.randn(
            size=(batch_size, self.action_dim),
            device=self.device,
            generator=rng
        )
        target_flow_actions = self.compute_flow_actions(states, noises).detach()
        actor_actions = self.one_step_actor(states, noises)
        distillation_loss = F.mse_loss(actor_actions, target_flow_actions)

        # q loss
        q_values = self.critic(states, actor_actions.clamp(self.min_action, self.max_action)).mean(0)
        q_loss = -q_values.mean()

        actor_loss = bc_flow_loss + self.bc_alpha * distillation_loss + q_loss

        self.actor_bc_flow_optimizer.zero_grad()
        self.one_step_actor_optimizer.zero_grad()

        actor_loss.backward()
        
        self.actor_bc_flow_optimizer.step()
        self.one_step_actor_optimizer.step()
        
        self.soft_critic_update()
        self.total_iterations += 1

        with torch.no_grad():
            policy_actions = self.sample_actions(states, rng)
            mse_actions = F.mse_loss(policy_actions, actions)
        
        return {
            "critic_loss": critic_loss.item(),
            "q_mean": q_values.mean().item(),
            "q_min": q_values.min().item(),
            "q_max": q_values.max().item(),
            "actor_loss": actor_loss.item(),
            "bc_flow_loss": bc_flow_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "mse_actions": mse_actions.item(),
        }

    def sample_actions(
            self,
            states: torch.Tensor,
            rng: torch.Generator | None = None
    ) -> torch.Tensor:
        '''
            sample actions from one-step policy
        '''
        batch_size = states.shape[0]

        noises = torch.randn(
            size=(batch_size, self.action_dim),
            generator=rng,
            device=self.device
        )
        actions = self.one_step_actor(states, noises)
        return actions.clamp(self.min_action, self.max_action)
    
    def compute_flow_actions(
            self,
            states: torch.Tensor,
            noises: torch.Tensor
    ) -> torch.Tensor:
        '''
            sample action from bc flow model
            using Euler Method
        '''
        batch_size = states.shape[0]
        h = 1 / self.num_flow_steps
        actions = noises

        for i in range(self.num_flow_steps):
            timesteps = torch.full(
                size=(batch_size, 1),
                fill_value=i * h,
                device=self.device
            )
            actions = actions + h * self.actor_bc_flow(states, actions, timesteps)

        return actions.clamp(self.min_action, self.max_action)

    def soft_critic_update(self) -> None:
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

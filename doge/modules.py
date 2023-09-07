from typing import Tuple, Optional
import torch
import numpy as np
from torch import nn
from math import sqrt


class DeterministicActor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 edac_init: bool = False,
                 max_action: float = 1.0) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        if edac_init:
            # init as in the EDAC paper
            for layer in self.trunk[::2]:
                nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.trunk(state)
        out = torch.tanh(out)
        return self.max_action * out
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state).cpu().numpy()
        return action


class EnsembledLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ensemble_size: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 0
        if fan_in > 0:
            bound = 1 / sqrt(fan_in)

        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight + self.bias
        return out


class EnsembledCritic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_critics: int = 2,
                 layer_norm: bool = True,
                 edac_init: bool = True) -> None:
        super().__init__()

        #block = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.num_critics = num_critics

        self.critic = nn.Sequential(
            EnsembledLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            EnsembledLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            EnsembledLinear(hidden_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            EnsembledLinear(hidden_dim, 1, num_critics)
        )

        if edac_init:
            # init as in the EDAC paper
            for layer in self.critic[::3]:
                nn.init.constant_(layer.bias, 0.1)

            nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([state, action], dim=-1)
        concat = concat.unsqueeze(0)
        concat = concat.repeat_interleave(self.num_critics, dim=0)
        q_values = self.critic(concat).squeeze(-1)
        return q_values


class Distance(nn.Module):
    # state_conditioned distance function
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 num_negative_samples: int,
                 num_in_policy_negative_samples: int = 10,
                 max_action: float = 1.0) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.energy_scale = torch.tensor(100.0)
        self.hidden_dim = hidden_dim

        self.num_negative_samples = num_negative_samples
        self.num_in_policy_negative_samples = int(num_negative_samples / 2 + num_in_policy_negative_samples)

        self.max_action = max_action
        self.action_range = 2 * max_action
        self.noise_action_range = self.action_range * 1.5  # taken from original implementation
        self.action_shift = 0.5

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def value(self,
              state: torch.Tensor,
              action: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([state, action], dim=-1)
        return self.net(concat)
    
    def linear_distance(self,
                        state: torch.Tensor,
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            returns tuple of tensors for a supervised learning in a manner (model_output, label (a.k.a desired output))
        '''
        batch_size = state.shape[0]

        states = state.unsqueeze(0).repeat(self.num_negative_samples, 1, 1)
        states = states.view(batch_size * self.num_negative_samples, self.state_dim)

        actions = action.unsqueeze(0).repeat(self.num_negative_samples, 1, 1)
        actions = actions.view(batch_size * self.num_negative_samples, self.action_dim)

        noise_actions = torch.rand([batch_size * self.num_negative_samples, self.action_dim]) - self.action_shift
        noise_actions = noise_actions.to(state.device) * self.noise_action_range

        difference = noise_actions - actions
        diff_norm = torch.norm(difference, dim=1, keepdim=True)

        output = self.value(states, noise_actions)
        return output, diff_norm
    
    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        positive_energy = (-self.value(state, action)).exp()

        states = state.unsqueeze(0).repeat(self.num_negative_samples, 1, 1)
        states = states.view(batch_size * self.num_negative_samples, self.state_dim)
        noise_actions = torch.rand([batch_size * self.num_negative_samples, self.action_dim]) - self.action_shift
        noise_actions = noise_actions.to(state.device) * self.noise_action_range

        negative_energy = (-self.value(states, noise_actions)).exp().view(self.num_negative_samples, batch_size, 1).sum(0)

        return positive_energy / (positive_energy + negative_energy)

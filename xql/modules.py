from math import sqrt
from typing import Optional, Tuple
import torch
from torch import nn
from torch.distributions import Normal, Distribution
import numpy as np


class Actor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 min_log_std: float = -5.0,
                 max_log_std: float = 2.0,
                 min_action: float = -1.0,
                 max_action: float = 1.0) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.min_action = min_action
        self.max_action = max_action
    
    def get_policy(self, state: torch.Tensor) -> Distribution:
        mean = self.net(state)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        
        policy = Normal(mean, log_std.exp())
        return policy
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self.get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob
    
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self.get_policy(state)

        #action = policy.mean
        if self.net.training:
            action = policy.sample()
        else:
            action = policy.mean
        
        return action[0].cpu().numpy()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self.get_policy(state)
        action = policy.rsample().clamp(self.min_action, self.max_action)

        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 layer_norm: bool = True,
                 edac_init: bool = True) -> None:
        super().__init__()

        #block = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        if edac_init:
            # init as in the EDAC paper
            for layer in self.critic[::3]:
                nn.init.constant_(layer.bias, 0.1)

            nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([state, action], dim=-1)
        q_values = self.critic(concat).squeeze(-1)  # shape: [batch_size,]
        return q_values


class ValueFunction(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 256,
                 layer_norm: bool = True) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


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

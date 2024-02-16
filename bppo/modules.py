from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Distribution


def orthogonal_init(m: nn.Module) -> None:
    for p in m.parameters():
        if len(p.size()) >= 2:
            nn.init.orthogonal_(p)


def log_prob(distribution: Distribution, sample: torch.Tensor) -> torch.Tensor:
    log_prob = distribution.log_prob(sample)
    if log_prob.ndim == 1:
        return log_prob
    return log_prob.sum(-1, keepdim=True)


class StochasticActor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 min_log_std: float = -5.0,
                 max_log_std: float = 0.0,
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
            nn.Linear(hidden_dim, 2 * action_dim),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.diff = max_log_std - min_log_std

        self.min_action = min_action
        self.max_action = max_action

        self.apply(orthogonal_init)
    
    def get_policy(self, state: torch.Tensor) -> Distribution:
        mean, log_std = self.net(state).chunk(2, dim=-1)
        # log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        log_std = self.min_log_std + 0.5 * self.diff * (log_std + 1)
        
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

        self.apply(orthogonal_init)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 layer_norm: bool = True) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
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

        self.apply(orthogonal_init)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([state, action], dim=-1)
        return self.net(concat)


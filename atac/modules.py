from typing import Optional, Tuple
import numpy as np
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, Distribution, TransformedDistribution, TanhTransform


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


class CholeskyActor(nn.Module):
    '''
        A Gaussian Actor that models Multivariate Normal distribution
        with learnable covariance matrix approximated by
        cholesky decomposition
    '''
    min_covariate_value: float = 1e-4

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 min_mu: float = -5.0,
                 max_mu: float = 5.0,
                 min_cov: float = -5.0,
                 max_cov: float = 20.0,
                 min_action: float = -1.0,
                 max_action: float = 1.0) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_action = max_action
        self.min_action = min_action
        self.min_mu = min_mu
        self.max_mu = max_mu
        self.min_cov = min_cov
        self.max_cov = max_cov

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.cholesky_layer = nn.Linear(hidden_dim, action_dim * (action_dim + 1) // 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.xavier_normal_(self.cholesky_layer.weight)
        nn.init.constant_(self.mu.bias, 0.0)
        nn.init.constant_(self.cholesky_layer.bias, 0.0)
    
    def get_tril(self, trunk_output: torch.Tensor) -> torch.Tensor:
        cholesky_vector = self.cholesky_layer(trunk_output).clamp(self.min_cov, self.max_cov)

        diag_index = torch.arange(self.action_dim, dtype=torch.long) + 1
        diag_index = torch.div(
            diag_index * (diag_index + 1), 2, rounding_mode="floor") - 1
        cholesky_vector[:, diag_index] = F.softplus(cholesky_vector[:, diag_index]) + self.min_covariate_value

        tril_indexes = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        cholesky = torch.zeros(size=(trunk_output.shape[0], self.action_dim, self.action_dim), dtype=torch.float32).to(trunk_output.device)
        cholesky[:, tril_indexes[0], tril_indexes[1]] = cholesky_vector
        return cholesky
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.trunk(state)

        mean = torch.sigmoid(self.mu(out).clamp(self.min_mu, self.max_mu))
        mean = self.min_action + (self.max_action - self.min_action) * mean
        cholesky = self.get_tril(out)

        policy = MultivariateNormal(mean, scale_tril=cholesky)
        action = policy.rsample()

        tanh_action = torch.tanh(action)

        log_prob = policy.log_prob(action).sum(-1)
        log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(-1)

        return tanh_action * self.max_action, log_prob


class StochasticActor(nn.Module):
    '''
        A Gaussian Actor that forward actions sampling from Normal distribution
        whose loc and scale parameters are state-dependent
    '''
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 min_log_std: float = -5.0,
                 max_log_std: float = 2.0,
                 min_action: float = -1.0,
                 max_action: float = 1.0,
                 edac_init: bool = True) -> None:
        super().__init__()

        self.action_dim = action_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        if edac_init:
            # init as in the EDAC paper
            for layer in self.trunk[::2]:
                nn.init.constant_(layer.bias, 0.1)

            nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
            nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
            nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
            nn.init.uniform_(self.log_std.bias, -1e-3, 1e-3)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.min_action = min_action
        self.max_action = max_action
    
    def get_policy(self, state: torch.Tensor) -> Distribution:
        hidden = self.trunk(state)
        mean, log_std = self.mu(hidden), self.log_std(hidden)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        
        policy = TransformedDistribution(
            Normal(mean, log_std.exp()),
            TanhTransform(cache_size=1)
        )

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

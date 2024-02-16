from math import sqrt
import torch
from torch import nn
from typing import Tuple, Optional
import numpy as np
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 edac_init: bool = True,
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
    
    def forward(self,
                state: torch.Tensor,
                deterministic: bool = False,
                need_log_prob: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_std = self.mu(hidden), self.log_std(hidden)

        log_std = torch.clip(log_std, -5, 2)
        policy_distribution = Normal(mu, torch.exp(log_std))

        if deterministic:
            action = mu
        else:
            action = policy_distribution.rsample()
        
        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            log_prob = policy_distribution.log_prob(action).sum(-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(-1)
            # shape [batch_size,]
        
        return tanh_action * self.max_action, log_prob
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
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
    
    def reset_parameters(self) -> None:
        scale_factor = sqrt(5)
        # default pytorch init
        for layer in range(self.ensemble_size):
            nn.init.kaiming_normal_(self.weight[layer], a=scale_factor)
        
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            x: [ensemble_size, batch_size, input_size]
            weight: [ensemble_size, input_size, out_size]
            bias: [ensemble_size, batch_size, out_size]
        '''
        return x @ self.weight + self.bias


class EnsembledCritic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hiddem_dim: int = 256,
                 num_critics: int = 10) -> None:
        super().__init__()

        self.critic = nn.Sequential(
            EnsembledLinear(state_dim + action_dim, hiddem_dim, num_critics),
            nn.ReLU(),
            EnsembledLinear(hiddem_dim, hiddem_dim, num_critics),
            nn.ReLU(),
            EnsembledLinear(hiddem_dim, hiddem_dim, num_critics),
            nn.ReLU(),
            EnsembledLinear(hiddem_dim, 1, num_critics)
        )

        self.reset_parameters()

        self.num_critics = num_critics
    
    def reset_parameters(self):
        for layer in self.critic[::2]:
            nn.init.constant_(layer.bias, 0.1)
        
        nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        concat = torch.cat([state, action], dim=-1)
        concat = concat.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)

        # [num_critics, batch_size]
        q_values = self.critic(concat).squeeze(-1)
        return q_values
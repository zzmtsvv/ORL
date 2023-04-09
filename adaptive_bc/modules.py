import torch
from torch import nn
import numpy as np


class Actor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 hidden_dim: int = 256) -> None:
        super().__init__()

        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.actor(state)
    
    @torch.no_grad()
    def act(self, state, device: str = "cpu") -> np.ndarray:
        state = state.reshape(1, -1)

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=device, dtype=torch.float32)
        
        return self(state).cpu().data.numpy().flatten()


class EnsembleLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ensemble_size: int) -> None:
        super().__init__()

        self.ensemble_size = ensemble_size
        scale_factor = 2 * in_features ** 0.5

        self.weight = nn.Parameter(torch.zeros(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_features))

        nn.init.trunc_normal_(self.weight, std=1 / scale_factor)
    
    def forward(self, x: torch.Tensor):

        if len(x.shape) == 2:
            #print(x.shape, self.weight.shape)
            x = torch.einsum('ij,bjk->bik', x, self.weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, self.weight)
        
        x = x + self.bias
        return x


class EnsembledCritic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_critics: int = 10) -> None:
        super().__init__()

        self.critics = nn.Sequential(
            EnsembleLinear(state_dim + action_dim, hidden_dim, ensemble_size=num_critics),
            nn.ReLU(),
            EnsembleLinear(hidden_dim, hidden_dim, ensemble_size=num_critics),
            nn.ReLU(),
            EnsembleLinear(hidden_dim, 1, ensemble_size=num_critics)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # shape: (num_critics, batch, 1)
        concat = torch.cat([state, action], 1)
        #print(f"concat shape {concat.shape}")

        #print(self.critics(concat).shape)
        return self.critics(concat)

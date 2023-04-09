import torch
from torch import nn


def init_weights_(m: nn.Module,
                  val: float = 3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-val, val)
        m.bias.data.uniform_(-val, val)


class Actor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = None,
                 dropout: float = None,
                 hidden_dim: int = 256,
                 uniform_initialization: bool = False) -> None:
        super().__init__()

        if dropout is None:
            dropout = 0
        self.max_action = max_action

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = self.actor(state)

        if self.max_action is not None:
            return self.max_action * torch.tanh(action)
        return action


class Critic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 uniform_initialization: bool = False) -> None:
        super().__init__()

        self.q1_ = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2_ = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor):
        concat = torch.cat([state, action], 1)

        return self.q1_(concat), self.q2_(concat)
    
    def q1(self,
           state: torch.Tensor,
           action: torch.Tensor) -> torch.Tensor:

        return self.q1_(torch.cat([state, action], 1))

from math import sqrt
import torch
from torch import nn


class AvgL1Norm(nn.Module):
    # class name is weird but i try to be consistent with the paper
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, eps: float = 1e-8):
        return x / x.abs().mean(dim=-1, keepdim=True).clamp_min(eps)


class TD7Encoder(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256,
                 activation: nn.Module = nn.ELU) -> None:
        super().__init__()

        self.f_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, embedding_dim),
            AvgL1Norm()
        )

        self.g_layers = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def f(self, states: torch.Tensor) -> torch.Tensor:
        return self.f_layers(states)
    
    def g(self,
          embeddings: torch.Tensor,
          actions: torch.Tensor) -> torch.Tensor:
        input_ = torch.cat([embeddings, actions], dim=-1)
        return self.g_layers(input_)


class TD7Actor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256,
                 activation: nn.Module = nn.ReLU) -> None:
        super().__init__()

        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            AvgL1Norm()
        )
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self,
                states: torch.Tensor,
                embeddings: torch.Tensor) -> torch.Tensor:
        out = self.state_layers(states)
        out = torch.cat([out, embeddings], dim=-1)
        return self.layers(out)
    
    def sample(self,
               states: torch.Tensor,
               embeddings: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.forward(states, embeddings))


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


class TD7Critic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256,
                 num_critics: int = 2,
                 activation: nn.Module = nn.ELU) -> None:
        super().__init__()

        self.num_critics = num_critics

        self.state_action_layers = nn.Sequential(
            EnsembledLinear(state_dim + action_dim, hidden_dim, ensemble_size=num_critics),
            AvgL1Norm()
        )
        self.layers = nn.Sequential(
            EnsembledLinear(2 * embedding_dim + hidden_dim, hidden_dim, ensemble_size=num_critics),
            activation(),
            EnsembledLinear(hidden_dim, hidden_dim, ensemble_size=num_critics),
            activation(),
            EnsembledLinear(hidden_dim, 1, ensemble_size=num_critics)
        )
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor,
                zsa: torch.Tensor,
                zs: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([states, actions], dim=-1)
        out = self.state_action_layers(state_action)
        out = torch.cat([
            out,
            zsa.repeat([self.num_critics] + [1] * len(zsa.shape)),
            zs.repeat([self.num_critics] + [1] * len(zs.shape))
        ], dim=-1)
        out = self.layers(out)
        return out

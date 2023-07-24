import numpy as np
from typing import Tuple, Optional
import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform


class AbstractPolicyClass(nn.Module):
    @staticmethod
    def init_weights(sequential: nn.Sequential,
                     orthogonal_init: bool) -> None:
        if orthogonal_init:
            for module in sequential[:-1]:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)
        
        if orthogonal_init:
            nn.init.orthogonal_(sequential[-1].weight, gain=0.01)
        else:
            nn.init.xavier_uniform_(sequential[-1].weight, gain=0.01)
        
        nn.init.constant_(sequential[-1].bias, 0.0)
    
    @staticmethod
    def extend_n_repeat(object: torch.Tensor,
                        dim: int,
                        num_repeats: int) -> torch.Tensor:
        return object.unsqueeze(dim).repeat_interleave(num_repeats, dim=dim)


class TanhGaussianWrapper(nn.Module):
    '''
        a functional class upon reparametrization trick
        with optional tanh transform (as one of variants for actor-critic training)

        tanh is used to constrain actor network output values within [-max_action, max_action] range.
    '''
    def __init__(self,
                 std_min: float = -20.0,
                 std_max: float = 2.0,
                 use_tanh: bool = True) -> None:
        super().__init__()

        self.std_min = std_min
        self.std_max = std_max
        self.use_tanh = use_tanh
    
    def forward(self,
                mean: torch.Tensor,
                log_std: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.std_min, self.std_max)
        std = torch.exp(log_std)

        if self.use_tanh:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        else:
            action_distribution = Normal(mean, std)
        
        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()
        
        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(AbstractPolicyClass):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 std_multiplier: float = 1.0,
                 std_offset: float = -1.0,
                 orthogonal_initialization: bool = False,
                 use_tanh: bool = True,
                 hidden_dim: int = 256) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_initialization
        self.use_tanh = use_tanh
        self.hidden_dim = hidden_dim

        self.log_std_multiplier = nn.Parameter(torch.tensor(std_multiplier, dtype=torch.float32))
        self.log_std_offset = nn.Parameter(torch.tensor(std_offset, dtype=torch.float32))
        self.tanh_gaussian = TanhGaussianWrapper(use_tanh=use_tanh)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)            
        )

        self.init_weights(self.net, orthogonal_initialization)
    
    def forward(self,
                states: torch.Tensor,
                deterministic: bool = False,
                num_repeats: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if num_repeats is not None:
            states = self.extend_n_repeat(states, 1, num_repeats)
        
        output = self.net(states)
        mean, log_std = torch.split(output, dim=-1)
        log_std = self.log_std_multiplier * log_std + self.log_std_offset

        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)

        return self.max_action * actions, log_probs
    
    def log_prob(self,
                 states: torch.Tensor,
                 actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3:
            states = self.extend_n_repeat(states, 1, actions.shape[1])
        out = self.net(states)

        mean, log_std = torch.split(out, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier * log_std + self.log_std_offset
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs
    
    @torch.no_grad()
    def act(self,
            state: np.ndarray,
            device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)

        actions, _ = self.forward(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedCritic(AbstractPolicyClass):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 orthogonal_initialization: bool = False,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 3) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_initialization

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.init_weights(self.net, orthogonal_initialization)
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        multiple_act = False
        
        if actions.ndim == 3 and states.ndim == 2:
            multiple_act = True

            states = self.extend_n_repeat(states, 1, actions.shape[1]).reshape(-1, states.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        
        forward_input = torch.cat([states, actions], dim=-1)
        q_values = torch.squeeze(self.net(forward_input), dim=-1)

        if multiple_act:
            q_values = q_values.reshape(batch_size, -1)
        
        return q_values
from math import sqrt
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
    '''
        trying a different variation of a stochastic actor in 
        comparison with actor from https://github.com/zzmtsvv/sac_rnd/blob/main/modules.py
    '''
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
        mean, log_std = torch.split(output, self.action_dim, dim=-1)
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
        multiple_actions = False
        batch_size = states.shape[0]
        if actions.ndim == 3 and states.ndim == 2:
            multiple_actions = True
            states = self.extend_n_repeat(states, 1, actions.shape[1]).reshape(
                -1, states.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([states, actions], dim=-1)
        q_values = torch.squeeze(self.net(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values



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
        return x @ self.weight + self.bias


class EnsembledCritic(AbstractPolicyClass):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_critics: int = 2,
                 layer_norm: bool = True,
                 orthogonal_init: bool = False) -> None:
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

        self.init_weights(self.critic, orthogonal_init)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        '''
            if multiple_actions, outputs q_values of shape [num_critics, batch_size, num_actions]
        '''
        multiple_actions = False
        batch_size = states.shape[0]

        if actions.ndim == 3 and states.ndim == 2:
            multiple_actions = True
            states = self.extend_n_repeat(states, 1, actions.shape[1]).reshape(-1, states.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        
        concat = torch.cat([states, actions], dim=-1)
        q_values = self.critic(concat).squeeze(-1)

        if multiple_actions:
            q_values = q_values.reshape(self.num_critics, batch_size, -1)

        return q_values


if __name__ == "__main__":
    action = torch.rand(16, 6)
    state = torch.rand(16, 17)

    critic = EnsembledCritic(17, 6)
    actor = TanhGaussianPolicy(17, 6, 1)
    q_func = FullyConnectedCritic(17, 6)

    action, log_prob = actor(state, num_repeats=10)
    print(action.shape, log_prob.shape)
    out = critic(state, action)
    print(out.shape)
    print((out - log_prob).shape)
    # out1, out2 = torch.split(out, 1, dim=0)

    # # print(out1)

    # print(torch.cat((state, action), dim=1).shape)
    # print(out.shape)
    # print(torch.maximum(out, out + 5).shape)
    # print()
    # q1 = q_func(state, action)
    # q2 = q_func(state, action)
    # torch.cat((q1, q2), dim=1)
    a = torch.rand(2, 64, 10)
    print(torch.logsumexp(a, dim=-1).shape)

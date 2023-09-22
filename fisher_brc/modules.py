from typing import Tuple, Optional
from math import sqrt
from typing import Optional, Tuple
import torch
from torch import nn
from torch import distributions
import numpy as np


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

        log_std = torch.clip(log_std, -20, 2)  # log_std = torch.clip(log_std, -5, 2) EDAC clipping
        policy_distribution = distributions.Normal(mu, torch.exp(log_std))

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


class MixtureGaussianActor(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_actors: int = 5,
                 min_log_std: float = -20.0,
                 max_log_std: float = 2.0,
                 min_action: float = -1.0,
                 max_action: float = 1.0) -> None:
        super().__init__()

        self.num_actors = num_actors
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.eps = 1e-6

        self.mixture_trunk = nn.Sequential(
            EnsembledLinear(state_dim, hidden_dim, num_actors),
            nn.ReLU(),
            EnsembledLinear(hidden_dim, hidden_dim, num_actors),
            nn.ReLU(),
            EnsembledLinear(hidden_dim, hidden_dim, num_actors),
            nn.ReLU()
        )

        self.logit_head = EnsembledLinear(hidden_dim, action_dim, num_actors)
        self.mu_head = EnsembledLinear(hidden_dim, action_dim, num_actors)
        self.log_std_head = EnsembledLinear(hidden_dim, action_dim, num_actors)
    
    def mixture_forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ = state.unsqueeze(0).repeat_interleave(self.num_actors, dim=0)
        out = self.mixture_trunk(input_)

        return self.logit_head(out), self.mu_head(out), self.log_std_head(out)
    
    def get_policy(self, state: torch.Tensor) -> distributions.Distribution:
        logits, mean, log_std = self.mixture_forward(state)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        batch_size = logits.shape[1]

        # reinterpreted_batch_ndims – the number of batch dims to reinterpret as event dims
        # the `num_actors` dim should be considered as event dim in Independent module,
        # so we are doing reshape
        logits = logits.reshape(batch_size, -1, self.num_actors)
        mean = mean.reshape(batch_size, -1, self.num_actors)
        log_std = log_std.reshape(batch_size, -1, self.num_actors)

        # print(logits.isnan().any(), mean.isnan().any(), log_std.isnan().any())

        components_distribution = distributions.TransformedDistribution(
            distributions.Normal(mean, log_std.exp()),
            distributions.TanhTransform(cache_size=1)
        )
        distribution = distributions.MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=logits),
            component_distribution=components_distribution
        )

        # I hope it works properly according to output shape
        return distributions.Independent(distribution, reinterpreted_batch_ndims=0)
    
    def log_prob(self,
                 states: torch.Tensor,
                 actions: torch.Tensor,
                 need_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        policy = self.get_policy(states)

        actions = actions.clamp(self.min_action + self.eps, self.max_action - self.eps)

        log_prob = policy.log_prob(actions).sum(-1, keepdim=True)

        entropy = None
        if need_entropy:
            sampled_actions = policy.sample()
            sampled_actions = sampled_actions.clamp(self.min_action + self.eps, self.max_action - self.eps)
            entropy = -policy.log_prob(sampled_actions).sum(-1, keepdim=True)

        return log_prob, entropy


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


if __name__ == "__main__":
    state = torch.rand(16, 17)
    action = torch.rand(16, 6)

    critic = EnsembledCritic(17, 6)
    actor = MixtureGaussianActor(17, 6)


    print(critic(state, action).min(0).values.shape)

    # print(critic(state, action).shape)
    # policy = actor.get_policy(state)
    # print(policy.sample().shape)
    # print(actor.log_prob(state, action, need_entropy=True)[1].shape)
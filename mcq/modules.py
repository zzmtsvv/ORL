from math import sqrt, log
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


class ConditionalVAE(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 latent_dim: Optional[int] = None,
                 max_action: float = 1.0,
                 hidden_dim: int = 750) -> None:
        super().__init__()

        if latent_dim is None:
            latent_dim = 2 * action_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action
        self.latent_dim = latent_dim
    
    def encode(self,
               state: torch.Tensor,
               action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(torch.cat([state, action], -1))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std
    
    def decode(self,
               state: torch.Tensor,
               z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(state.device).clamp(-0.5, 0.5)
        
        x = torch.cat([state, z], -1)
        return self.max_action * self.decoder(x)
    
    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)

        return self.decode(state, z), mean, std
    
    def importance_sampling_loss(self,
                                 state: torch.Tensor,
                                 action: torch.Tensor,
                                 beta: float,
                                 num_samples: int = 10) -> torch.Tensor:
        mean, std = self.encode(state, action)
        
        mean = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean + std * torch.randn_like(std)
        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        
        mean_decoded = self.decode(state, z)
        scale_factor = sqrt(beta) / 2

        log_prob_q_zx = Normal(loc=mean, scale=std).log_prob(z)
        mean_prior = torch.zeros_like(z).to(state.device)
        std_prior = torch.ones_like(z).to(state.device)
        log_prob_p_z = Normal(loc=mean_prior, scale=std_prior).log_prob(z)
        std_decoded = torch.ones_like(mean_decoded).to(state.device) * scale_factor
        log_prob_p_xz = Normal(loc=mean_decoded, scale=std_decoded).log_prob(action)

        w = log_prob_p_xz.sum(-1) + log_prob_p_z.sum(-1) - log_prob_q_zx.sum(-1)
        score = w.logsumexp(dim=-1) - log(num_samples)
        return -score
    
    def elbo_loss(self,
                  state: torch.Tensor,
                  action: torch.Tensor,
                  beta: float = 1,
                  num_samples: int = 10) -> torch.Tensor:
        mean, std = self.encode(state, action)

        mean = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean + std * torch.randn_like(std)
        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)

        decoded = self.decode(state, z)
        reconstruction_loss = (decoded - action).pow(2).mean(dim=(1, 2))

        kl_loss = -1 / 2 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        loss = reconstruction_loss + beta * kl_loss
        return loss
    
    def decode_multiple(self,
                        state: torch.Tensor,
                        z: Optional[torch.Tensor] = None,
                        num_samples: int = 10) -> torch.Tensor:
        if z is None:
            z = torch.randn(
                (num_samples, state.shape[0], self.latent_dim)
                ).to(state.device).clamp(-0.5, 0.5)

        state = state.repeat(num_samples, 1, 1)
        
        return self.max_action * self.decoder(torch.cat([state, z], dim=-1))


if __name__ == "__main__":
    vae = ConditionalVAE(17, 6)
    state = torch.rand(4, 17)
    action = torch.rand(4, 6)

    print(vae.decode(state).shape)


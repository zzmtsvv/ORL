import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from typing import Tuple
from math import sqrt, log


class ConditionalVAE(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 latent_dim: int,
                 max_action: int = None,
                 hidden_dim: int = 750,
                 device: torch.device = None) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_action = max_action
        self.latent_dim = latent_dim

        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)
    
    def encode(self,
               state: torch.Tensor,
               action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        z = F.relu(self.e1(torch.cat([state, action], -1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        std = torch.exp(self.log_std(z).clamp(-4, 15))  # see __ in 'paper' folder
        return mean, std
    
    def decode(self,
               state: torch.Tensor,
               z: torch.Tensor = None) -> torch.Tensor:
        
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)  # see __ in 'paper' folder
        
        action = F.relu(self.d1(torch.cat([state, z], -1)))
        action = F.relu(self.d2(action))
        action = self.d3(action)

        if self.max_action is not None:
            return self.max_action * torch.tanh(action)
        return action
    
    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor):
        
        mean, std = self.encode(state, action)
        z = mean * std * torch.randn_like(std)

        return self.decode(state, z), mean, std
    
    def importance_sampling_loss(self,
                                 state: torch.Tensor,
                                 action: torch.Tensor,
                                 beta: float,
                                 num_samples: int = 10) -> torch.Tensor:
        # see eq8 in 'paper' folder
        mean, std = self.encode(state, action)
        
        mean = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean + std * torch.randn_like(std)
        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        
        mean_decoded = self.decode(state, z)
        scale_factor = sqrt(beta) / 2

        log_prob_q_zx = Normal(loc=mean, scale=std).log_prob(z)
        mean_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_prob_p_z = Normal(loc=mean_prior, scale=std_prior).log_prob(z)
        std_decoded = torch.ones_like(mean_decoded).to(self.device) * scale_factor
        log_prob_p_xz = Normal(loc=mean_decoded, scale=std_decoded).log_prob(action)

        w = log_prob_p_xz.sum(-1) + log_prob_p_z.sum(-1) - log_prob_q_zx.sum(-1)
        score = w.logsumexp(dim=-1) - log(num_samples)
        return -score
    
    def elbo_loss(self,
                  state: torch.Tensor,
                  action: torch.Tensor,
                  beta: float,
                  num_samples: int = 10) -> torch.Tensor:
        # see eq7 in 'paper' folder
        mean, std = self.encode(state, action)

        mean = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean + std * torch.randn_like(std)
        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)

        decoded = self.decode(state, z)
        reconstruction_loss = ((decoded - action) ** 2).mean(dim=(1, 2))

        kl_loss = -1 / 2 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        loss = reconstruction_loss + beta * kl_loss
        return loss
    
    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))

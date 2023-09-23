# mostly taken from official implementation https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
# and hugging face tutorial on DDPM for Computer Vision
from typing import Union, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def vp_beta_schedule(timesteps: int, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


def extract(a: torch.Tensor,
            t: torch.Tensor,
            x_shape) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, time: torch.Tensor):
        device = time.device
        half = self.embedding_dim // 2
        
        embeddings = np.log(10000) / (half - 1)
        embeddings = torch.exp(torch.arange(half, device=device) * (-1) * embeddings)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class MLP(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 time_dim: int = 16) -> None:
        super().__init__()

        self.time_net = nn.Sequential(
            PositionalEncoding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + time_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self,
               action: torch.Tensor,
               state: torch.Tensor,
               time: torch.Tensor) -> torch.Tensor:
        time_features = self.time_net(time)
        net_input = torch.cat([action, state, time_features], dim=-1)

        return self.net(net_input)


class WeightedL2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                weights: Union[torch.Tensor, float] = 1.0) -> torch.Tensor:
        loss = F.mse_loss(prediction, target, reduction="none")
        return (loss * weights).mean()


class DDPM(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 trunk: MLP,
                 max_action: float = 1.0,
                 num_timesteps: int = 100,) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trunk = trunk
        self.max_action = max_action
        self.num_timesteps = num_timesteps

        betas = vp_beta_schedule(num_timesteps)
        self.betas = betas
        self.alphas = 1.0 - betas

        self.alphas_cumprod: torch.Tensor = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        self.sqrt_inverted_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_inverted_minus_one_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.log_one_minus_alphas_cumprod = torch.log(1. / self.alphas_cumprod)
        
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean2 = (1. - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        self.loss_fn = WeightedL2()
    
    def forward_from_noise(self,
                           a_t: torch.Tensor,
                           timestep: torch.Tensor,
                           noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_inverted_alphas_cumprod, timestep, a_t.shape) * a_t - \
               extract(self.sqrt_inverted_minus_one_alphas_cumprod, timestep, a_t.shape) * noise
    
    def q_posterior(self,
                    a_start: torch.Tensor,
                    a_t: torch.Tensor,
                    timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = extract(self.posterior_mean1, timestep, a_t.shape) * a_start + \
               extract(self.posterior_mean2, timestep, a_t.shape) * a_t
        
        variance = extract(self.posterior_variance, timestep, a_t.shape)
        log_variance = extract(self.posterior_log_variance, timestep, a_t.shape)
        return mean, variance, log_variance
    
    def p(self,
          action: torch.Tensor,
          state: torch.Tensor,
          timestep: torch.Tensor):
        reconstructed = self.forward_from_noise(action, timestep, self.trunk(action, state, timestep))
        reconstructed = reconstructed.clamp(-self.max_action, self.max_action)

        return self.q_posterior(reconstructed, action, timestep)
    
    def p_sample(self,
                 action: torch.Tensor,
                 state: torch.Tensor,
                 timestep: torch.Tensor) -> torch.Tensor:
        batch_size = action.shape[0]
        
        model_mean, _, model_log_variance = self.p(action, state, timestep)
        noise = torch.randn_like(action)

        mask = (1. - (timestep == 0).float()).reshape(batch_size, *((1,) * (len(action.shape) - 1)))

        return model_mean + mask * (model_log_variance / 2).exp() * noise
    
    def p_sample_loop(self, state: torch.Tensor) -> torch.Tensor:
        device = state.device
        batch_size = state.shape[0]

        action = torch.randn((batch_size, self.action_dim), device=device)

        for i in reversed(range(self.num_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            action = self.p_sample(action, state, timesteps)
        
        return action
    
    def sample(self, state: torch.Tensor) -> torch.Tensor:
        action = self.p_sample_loop(state)
        return action.clamp(-self.max_action, self.max_action)
    
    def q_sample(self,
                 a_start: torch.Tensor,
                 timestep: torch.Tensor,
                 noise: torch.Tensor):
        return extract(self.sqrt_alphas_cumprod, timestep, a_start.shape) * a_start + \
               extract(self.sqrt_one_minus_alphas_cumprod, timestep, a_start.shape) * noise
    
    def p_loss(self,
               a_start: torch.Tensor,
               state: torch.Tensor,
               timestep: torch.Tensor,
               weights: Union[torch.Tensor, float] = 1.0) -> torch.Tensor:
        noise = torch.randn_like(a_start)

        a_noisy = self.q_sample(a_start, timestep, noise)
        reconstructed = self.trunk(a_noisy, state, timestep)

        assert noise.shape == reconstructed.shape

        loss = self.loss_fn(reconstructed, noise, weights)
        return loss
    
    def loss(self,
             action: torch.Tensor,
             state: torch.Tensor,
             weights: Union[torch.Tensor, float] = 1.0) -> torch.Tensor:
        batch_size = action.shape[0]

        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=action.device).long()
        return self.p_loss(action, state, timesteps, weights)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.sample(state)


if __name__ == "__main__":
    mlp = MLP(17, 6)
    actor = DDPM(17, 6, mlp, num_timesteps=5)

    state = torch.rand(32, 17)

    print(actor(state).shape)

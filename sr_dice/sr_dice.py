from copy import deepcopy
import torch
from torch.nn import functional as F
from modules import AutoEncoder, Critic, DeterministicActor


class SR_DICE:
    def __init__(self,
                 device: str,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 max_action: float = 1.0,
                 discount: float = 0.99,
                 tau: float = 5e-3) -> None:
        self.device = device
        
        self.autoencoder = AutoEncoder(state_dim, action_dim, hidden_dim).to(device)
        self.ae_optim = torch.optim.Adam(self.autoencoder.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.weights = torch.ones(1, hidden_dim, requires_grad=True, device=device)
        self.weights_optimizer = torch.optim.Adam([self.weights], lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.max_action = max_action

        self.total_iterations = 0
    
    def train_autoencoder(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor) -> None:
        recon_next_states, recon_rewards, recon_actions, latents = self.autoencoder(states, actions)

        loss = F.mse_loss(recon_next_states, next_states) + \
               0.1 * F.mse_loss(recon_rewards, rewards) + \
               F.mse_loss(recon_actions, actions)
        
        self.ae_optim.zero_grad()
        loss.backward()
        self.ae_optim.step()
    
    def train_succ_repr(self,
                        policy: DeterministicActor,
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        next_states: torch.Tensor,
                        dones: torch.Tensor) -> None:
        with torch.no_grad():
            next_action = policy(next_states)
            next_action = (next_action + torch.randn_like(next_action) * self.max_action * 0.1).clamp(-self.max_action, self.max_action)
            
            latent = self.autoencoder.latent(states, actions)
            target_Q = latent + self.discount * (1.0 - dones) * self.critic_target(next_states, next_action)
        
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.soft_critic_update()

    def soft_critic_update(self) -> None:
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)
    
    def train_off_policy_eval(self,
                        policy: DeterministicActor,
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_states: torch.Tensor,
                        dones: torch.Tensor) -> None:
        pass


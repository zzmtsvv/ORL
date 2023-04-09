import torch
from torch.nn import functional as F
from copy import deepcopy
import numpy as np

try:
    from modules import Actor, EnsembledCritic
except ModuleNotFoundError:
    from .modules import Actor, EnsembledCritic


class RandomizedEnsembles_BC:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 discount_factor: float = 0.99,
                 tau: float = 0.005,
                 exploration_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_frequency: int = 2,
                 num_q_networks: int = 10,
                 alpha_finetune: float = 0.4,
                 pretrain: bool = False,
                 minimize_over_q: bool = False,
                 Kp: float = 0.00003,
                 Kd: float = 0.0001) -> None:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = EnsembledCritic(state_dim, action_dim, num_critics=num_q_networks).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount_factor = discount_factor
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_frequency
        self.num_nets = num_q_networks
        self.alpha = alpha_finetune
        self.alpha_finetune = alpha_finetune
        self.pretrain = pretrain
        self.minimize_over_q = minimize_over_q
        self.kp = Kp
        self.kd = Kd
    
    def update_alpha(self,
                     episode_timesteps,
                     average_return,
                     current_return,
                     target_return: float = 1.05) -> None:
        # see eq2.PNG in 'paper' folder
        self.alpha += episode_timesteps * (self.kp * (average_return - target_return) + self.kd * max(0, average_return - current_return))
        self.alpha = max(0.0, min(self.alpha, self.alpha_finetune))
    
    def train(self, data):
        self.iteration = 1

        state, action, reward, next_state, done = data

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.exploration_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            if self.minimize_over_q and not self.pretrain:
                print(f"input shape: {next_state.shape}, {next_action.shape}")
                tgt_qs = self.critic_target(next_state, next_action)
                tgt_q, _ = torch.min(tgt_qs, dim=0)
            else:  # REDQ
                random_indexes = np.random.permutation(self.num_nets)
                tgt_qs = self.critic_target(next_state, next_action)[random_indexes]
                tgt_q1, tgt_q2 = tgt_qs[:2]
                tgt_q = torch.min(tgt_q1, tgt_q2)
            
            tgt_q = reward + (1 - done) * self.discount_factor * tgt_q
        
        current_qs = self.critic(state, action)

        critic_loss = F.mse_loss(current_qs.unsqueeze(0), tgt_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # policy update
        if not self.iteration % self.policy_freq:

            pi = self.actor(state)
            q = self.critic(state, pi).mean(0)
            
            actor_loss = -q.mean() / q.abs().mean().detach() + self.alpha * F.mse_loss(pi, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update("actor")
            self.soft_update("critic")

        return {
            "critic_loss": critic_loss.item(),
            "critic_Qs": current_qs[0].mean().item()}

    
    def act(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        if isinstance(state, np.ndarray):
            state = self.to_tensor(state, device=self.device)
        else:
            state = state.to(self.device)
        
        with torch.no_grad():
            action = self.actor(state)
        
        return action.cpu().data.numpy().flatten()
    
    def soft_update(self, regime):
        if regime == "actor":
            for param, tgt_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
        else:
            for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
    
    @staticmethod
    def to_tensor(data, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return torch.tensor(data, dtype=torch.float32, device=device)
    
    def save(self, filename):
        torch.save({
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict()
        }, filename + '_policy.pth')

    def load(self, filename):
        policy_dict = torch.load(filename + "_policy.pth")

        self.critic.load_state_dict(policy_dict["critic"])
        self.critic_optimizer.load_state_dict(policy_dict["critic_optimizer"])
        self.critic_target = deepcopy(self.critic)

        self.actor.load_state_dict(policy_dict["actor"])
        self.actor_optimizer.load_state_dict(policy_dict["actor_optimizer"])
        self.actor_target = deepcopy(self.actor)

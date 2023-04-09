import os
import torch
from torch.nn import functional as F
from copy import deepcopy
import numpy as np

try:
    from utils import make_dir
    from logger import Logger
    from vae import ConditionalVAE
    from modules import Actor, Critic
    from dataset import ReplayBuffer
except ModuleNotFoundError:
    from .utils import make_dir
    from .logger import Logger
    from .vae import ConditionalVAE
    from .modules import Actor, Critic
    from .dataset import ReplayBuffer


class SPOT:
    diverging_threshold = 1e4

    def __init__(self,
                 vae: ConditionalVAE,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = None,
                 discount_factor: float = 0.99,
                 tau: float = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_frequency: int = 2,
                 beta: float = 0.5,
                 lambda_: float = 1.0,
                 lr: float = 3e-4,
                 actor_lr: float = None,
                 with_q_norm: bool = True,
                 num_samples: int = 1,
                 use_importance_sampling: bool = False,
                 actor_hidden_dim: int = 256,
                 critic_hidden_dim: int = 256,
                 actor_dropout: float = 0.1,
                 actor_init_w: bool = False,
                 critic_init_w: bool = False,
                 lambda_cool: bool = False,
                 lambda_end: float = 0.2) -> None:
        
        self.iterations = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action, actor_dropout, actor_hidden_dim, actor_init_w).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr or actor_lr)

        self.critic = Critic(state_dim, action_dim, critic_hidden_dim, critic_init_w).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount_factor = discount_factor
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency
        self.vae = vae
        self.beta = beta
        self.num_samples = num_samples
        self.use_importance_sampling = use_importance_sampling
        self.with_q_norm = with_q_norm
        self.lambda_ = lambda_
        self.lambda_cool = lambda_cool
        self.lambda_end = lambda_end
    
    @staticmethod
    def to_tensor(data, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(data, dtype=torch.float32, device=device)
    
    @torch.no_grad()
    def act(self, state: np.ndarray) -> np.ndarray:
        self.actor.eval()
        state = self.to_tensor(state.reshape(1, -1), device=self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        return action

    def soft_update(self, regime):
        if regime == "actor":
            for param, tgt_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
        else:
            for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
    
    def train(self,
              replay_buffer: ReplayBuffer,
              batch_size: int = 256,
              logger: Logger = None) -> None:
        self.iterations += 1
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            tgt_q1, tgt_q2 = self.critic_target(next_state, next_action)
            tgt_q = torch.min(tgt_q1, tgt_q2)

            tgt_q = reward + (1 - done) * self.discount_factor * tgt_q  # eq1 in 'paper' folder
        
        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, tgt_q) + F.mse_loss(current_q2, tgt_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if logger is not None:
            logger.log("train/critic_loss", critic_loss, self.iterations)
        
        if not self.iterations % self.policy_frequency:
            
            pi = self.actor(state)
            q = self.critic.q1(state, pi)
            
            if self.use_importance_sampling:
                density_estimator_loss = self.vae.importance_sampling_loss(state, pi, self.beta, self.num_samples)
            else:
                density_estimator_loss = self.vae.elbo_loss(state, pi, self.beta, self.num_samples)
            
            # see practical_algo.jpeg in 'paper' folder
            if self.with_q_norm:
                actor_loss = -q.mean() / q.abs().mean().detach() + self.lambda_ * density_estimator_loss.mean()
            else:
                actor_loss = -q.mean() + self.lambda_ * density_estimator_loss.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if logger is not None:
                logger.log("train/Q", q.mean(), self.iterations)
                logger.log("train/actor_loss", actor_loss, self.iterations)
                logger.log("train/neg_log_beta", density_estimator_loss.mean(), self.iterations)
                logger.log("train/neg_log_beta_max", density_estimator_loss.max(), self.iterations)
            
            if q.mean().item() > self.diverging_threshold:
                exit()
            
            self.soft_update(regime="actor")
            self.soft_update(regime="critic")

    def train_online(self,
                     replay_buffer: ReplayBuffer,
                     batch_size: int = 256,
                     logger: Logger =None) -> None:
        self.iterations += 1
        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            tgt_q1, tgt_q2 = self.critic_target(next_state, next_action)
            tgt_q = torch.min(tgt_q1, tgt_q2)

            tgt_q = reward + (1 - done) * self.discount_factor * tgt_q
        
        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, tgt_q) + F.mse_loss(current_q2, tgt_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if logger is not None:
            logger.log("train/critic_loss", critic_loss, self.iterations)
        
        if not self.iterations % self.policy_frequency:
            
            pi = self.actor(state)
            q = self.critic.q1(state, pi)
            
            if self.use_importance_sampling:
                density_estimator_loss = self.vae.importance_sampling_loss(state, pi, self.beta, self.num_samples)
            else:
                density_estimator_loss = self.vae.elbo_loss(state, pi, self.beta, self.num_samples)
            
            # additional component for online learning
            lambda_ = self.lambda_
            if self.lambda_cool:
                lambda_ = self.lambda_ * max(self.lambda_end, (1.0 - self.iterations / 1000000))

                if logger is not None:
                    logger.log("train/lambda_", lambda_, self.iterations)
            
            if self.with_q_norm:
                actor_loss = -q.mean() / q.abs().mean().detach() + lambda_ * density_estimator_loss.mean()
            else:
                actor_loss = -q.mean() + lambda_ * density_estimator_loss.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if logger is not None:
                logger.log("train/Q", q.mean(), self.iterations)
                logger.log("train/actor_loss", actor_loss, self.iterations)
                logger.log("train/neg_log_beta", density_estimator_loss.mean(), self.iterations)
                logger.log("train/neg_log_beta_max", density_estimator_loss.max(), self.iterations)
            

            self.soft_update(regime="actor")
            self.soft_update(regime="critic")

    def save(self, model_dir):
        make_dir(model_dir)

        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(self.iterations)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(self.iterations)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            model_dir, f"critic_optimizer_s{str(self.iterations)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(self.iterations)}.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(model_dir, f"actor_target_s{str(self.iterations)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            model_dir, f"actor_optimizer_s{str(self.iterations)}.pth"))

    def load(self, model_dir, step=1000000):
        self.critic.load_state_dict(torch.load(os.path.join(model_dir, f"critic_s{str(step)}.pth")))
        self.critic_target.load_state_dict(torch.load(os.path.join(model_dir, f"critic_target_s{str(step)}.pth")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(model_dir, f"critic_optimizer_s{str(step)}.pth")))

        self.actor.load_state_dict(torch.load(os.path.join(model_dir, f"actor_s{str(step)}.pth")))
        self.actor_target.load_state_dict(torch.load(os.path.join(model_dir, f"actor_target_s{str(step)}.pth")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(model_dir, f"actor_optimizer_s{str(step)}.pth")))

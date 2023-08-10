import torch
from tqdm import trange
from typing import Dict, Tuple, Union
import numpy as np
from config import calql_config
from cal_ql import CalQL
from dataset import ReplayBuffer
from utils import seed_everything
import wandb

import gym


_Number = Union[float, int]


class CALQLTrainer:
    def __init__(self,
                 cfg: calql_config) -> None:
        self.device = cfg.device
        self.cfg = cfg

        self.env = gym.make(cfg.env)
        
        self.eval_env = gym.make(cfg.env)
        self.eval_env.seed(cfg.eval_seed)
        self.eval_env.action_space.seed(cfg.eval_seed)

        seed_everything(cfg.seed, self.env)

        self.batch_size_offline = int(cfg.batch_size * cfg.mixing_ratio)
        self.batch_size_online = cfg.batch_size - self.batch_size_offline

        self.max_steps = cfg.max_episode_steps

        self.offline_buffer = ReplayBuffer(cfg.state_dim,
                                           cfg.action_dim,
                                           cfg.buffer_size)
        self.state_mean, self.state_std = self.offline_buffer.normalize_states()
        
        self.online_buffer = ReplayBuffer(cfg.state_dim,
                                           cfg.action_dim,
                                           cfg.buffer_size)
        
        self.offline_buffer.from_json(cfg.env)

        self.cal_ql = CalQL(cfg)

    def offline_train(self):
        run_name = f"{self.cfg.env}_{self.cfg.seed}"
        print(f"offline pretraining on {self.device}")

        with wandb.init(project=self.cfg.project, group=f"offline_{self.cfg.group}", name=run_name, job_type="offline_training"):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})
            
            for t in trange(self.cfg.offline_iterations):
                batch = self.offline_buffer.sample(self.cfg.batch_size)
                states, actions, rewards, next_states, dones, mc_returns = [b.to(self.device) for b in batch]

                logging_dict = self.cal_ql.train(states,
                                                actions,
                                                rewards,
                                                next_states,
                                                dones,
                                                mc_returns)
                
                wandb.log(logging_dict, step=self.cal_ql.total_iterations)

    def online_train(self):
        run_name = f"{self.cfg.env}_{self.cfg.seed}"
        print(f"online tuning on {self.device}")
        
        self.cal_ql.switch_calibration()
        self.cal_ql.cfg.alpha = self.cfg.alpha_online

        with wandb.init(project=self.cfg.project, group=f"online_{self.cfg.group}", name=run_name, job_type="online_training"):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            state, done = self.env.reset(), False
            episode_return = 0
            episode_step = 0
            
            for t in trange(self.cfg.online_iterations):
                episode_step += 1
                action, _ = self.cal_ql.actor(
                    torch.tensor(
                    state.reshape(1, -1),
                    device=self.device,
                    dtype=torch.float32
                    )
                )
                action = action.cpu().data.numpy().flatten()
                raise NotImplementedError()
    
    def save(self):
        torch.save(self.cal_ql.state_dict(), self.cfg.checkpoints_path)
    
    def load(self):
        state_dict = torch.load(self.cfg.checkpoints_path)
        self.cal_ql.load_state_dict(state_dict)

    def modify_reward(self,
                      dataset: Dict[str, np.ndarray],
                      max_episode_steps: int = 1000) -> Tuple[Dict[str, np.ndarray], Dict[str, _Number]]:

        min_return, max_return = self.reward_range(dataset, max_episode_steps)
        # minimax scaling
        dataset["rewards"] /= max_return - min_return
        dataset["rewards"] *= max_episode_steps
        data = {
            "max_return": max_return,
            "min_return": min_return,
            "max_episode_steps": max_episode_steps
        }
        dataset["rewards"] = dataset["rewards"] * self.cfg.reward_scale + self.cfg.reward_bias

        return dataset, data

    def modify_reward_online(self,
                             reward: float,
                             modification_data: Dict[str, _Number]):
        reward /= modification_data["max_return"] - modification_data["min_return"]
        reward *= modification_data["max_episode_steps"]

        reward = reward * self.cfg.reward_scale + self.cfg.reward_bias
        return reward

    def reward_range(self,
                     dataset: Dict[str, np.ndarray],
                     max_episode_steps: int) -> Tuple[float, float]:
        returns, lengths = [], []
        episode_return, episode_length = 0, 0

        for reward, done in zip(dataset["rewards"], dataset["terminals"]):
            episode_return += float(reward)
            episode_length += 1

            if done or episode_length == max_episode_steps:
                returns.append(episode_return)
                lengths.append(episode_length)

                episode_return, episode_length = 0, 0
        
        lengths.append(episode_length)
        assert sum(lengths) == len(dataset["rewards"])
        return min(returns), max(returns)

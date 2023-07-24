import torch
import numpy as np
import os
import gym
from dataset import ReplayBuffer
from typing import Optional, Tuple
import json
from imageio import mimsave
import random


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def seed_everything(seed: int,
                    env: Optional[gym.Env] = None,
                    use_deterministic_algos: bool = False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic_algos)
    random.seed(seed)


def evaluate_policy(policy,
                    env_name,
                    seed,
                    eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 42)

    average_reward, average_length = 0, 0

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            action = policy.act(state)
            state, reward, done, _ = eval_env.step(action)
            average_reward += reward
            average_length += 1
        
        average_reward /= eval_episodes
        average_length = int(average_length / eval_episodes)

        d4rl_score = eval_env.get_normalized_score(average_reward) * 100

        return {
            "d4rl": d4rl_score,
            "evaluation": average_reward,
            "length": average_length
        }


def rollout(batch_size,
            horizon,
            transition,
            policy,
            env_buffer,
            model_buffer,
            exploration_noise=0.1,
            max_action=1):
    raise NotImplementedError()


def parse_json_dataset(filename: str) -> Tuple[int, int, float]:
    max_action = 1.0

    if not filename.endswith('.json'):
        filename = filename + '.json'

    filename_ = os.path.join("json_datasets", filename)
    with open(filename_) as f:
        obj = json.load(f)
    
    states = np.array(obj["observations"])
    actions = np.array(obj["actions"])

    return states.shape[1], actions.shape[1], max_action


class VideoRecorder:
    def __init__(self, dir_name, height=512, width=512, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env:gym.Env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                # camera_id=self.camera_id
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            mimsave(path, self.frames, fps=self.fps)
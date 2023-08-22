import os
import torch
import random
from tqdm import trange
import pickle
from typing import Tuple, DefaultDict, Dict, List
from collections import defaultdict
import numpy as np


def discounted_cumulative_sum(x: np.ndarray,
                              gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]

    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_trajectories(filename: str,
                      gamma: float = 1.0) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, np.ndarray]]:
    dataset = dict()

    with open(f"pickle_datasets/{filename}.pkl", "rb") as f:
        meow: Dict[str, np.ndarray] = pickle.load(f)

        for k, v in meow.items():
            if k in ("observations", "actions", "next_observations", "rewards", "terminals", "timeouts"):
                v = np.array(v)
                if k != "terminals":
                    v = v.astype(np.float32)
                
                dataset[k] = v
    
    trajectory, trajectory_length = [], []
    data = defaultdict(list)

    for i in trange(dataset["rewards"].shape[0], desc="loading trajectories"):
        data["observations"].append(dataset["observations"][i])
        data["actions"].append(dataset["actions"][i])
        data["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data.items()}

            episode_data["returns"] = discounted_cumulative_sum(episode_data["rewards"], gamma=gamma)

            trajectory.append(episode_data)
            trajectory_length.append(episode_data["actions"].shape[0])

            data = defaultdict(list)
    
    info = {
        "state_mean": dataset["observations"].mean(0, keepdims=True),
        "state_std": dataset["observations"].std(0, keepdims=True) + 1e-8,
        "trajectory_lengths": np.array(trajectory_length),
    }

    return trajectory, info


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def seed_everything(seed: int,
                    use_deterministic_algos: bool = False):
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic_algos)
    random.seed(seed)


if __name__ == "__main__":
    print(torch.__version__)

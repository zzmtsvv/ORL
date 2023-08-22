import random
from typing import Dict, Iterator, Tuple, DefaultDict
import numpy as np
from torch.utils.data import IterableDataset


class SequenceDataset(IterableDataset):
    def __init__(self,
                 dataset: DefaultDict[str, np.ndarray],
                 info: Dict[str, np.ndarray],
                 sequence_length: int = 10,
                 reward_scale: float = 1.0) -> None:
        super().__init__()

        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.reward_scale = reward_scale
        self.sequence_length = sequence_length

        self.state_mean = info["state_mean"]
        self.state_std = info["state_std"]
        self.p = info["trajectory_lengths"] / info["trajectory_lengths"].sum()
    
    def sample2yield(self,
                     start_index: int,
                     trajectory_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        traj = self.dataset[trajectory_index]

        states: np.ndarray = traj["observations"][start_index:start_index + self.sequence_length]
        actions: np.ndarray = traj["actions"][start_index:start_index + self.sequence_length]
        returns: np.ndarray = traj["returns"][start_index:start_index + self.sequence_length]
        time_steps = np.arange(start_index, start_index + self.sequence_length)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale

        mask = np.hstack([np.ones(states.shape[0]), np.zeros(self.sequence_length - states.shape[0])])

        if states.shape[0] < self.sequence_length:
            states = self.pad(states, desired_size=self.sequence_length)
            actions = self.pad(actions, desired_size=self.sequence_length)
            returns = self.pad(returns, desired_size=self.sequence_length)
        
        return states, actions, returns, time_steps, mask
    
    def __iter__(self) -> Iterator:
        while True:
            trajectory_index = np.random.choice(self.dataset_len, p=self.p)
            start_idx = random.randint(0, self.dataset[trajectory_index]["rewards"].shape[0] - 1)

            yield self.sample2yield(start_idx, trajectory_index)
    
    @staticmethod
    def pad(input_: np.ndarray,
            desired_size: int,
            axis: int = 0,
            constant_values: float = 0.0) -> np.ndarray:
        pad_size = desired_size - input_.shape[axis]
        if pad_size <= 0:
            return input_
        
        pad_width = [(0, 0)] * input_.ndim
        pad_width[axis] = (0, pad_size)
        return np.pad(input_, pad_width=pad_width, mode="constant", constant_values=constant_values)

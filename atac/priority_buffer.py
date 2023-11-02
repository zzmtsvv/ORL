# https://arxiv.org/abs/1511.05952
import torch
from typing import List, Union, Tuple
from random import uniform
import numpy as np
from buffer import ReplayBuffer


_Number = Union[int, float, bool]


class HeapSum:
    def __init__(self, capacity: int) -> None:
        self.nodes = [0] * (2 * capacity - 1)
        self.data = [None] * capacity

        self.capacity = capacity
        self.pointer = 0
        self.size = 0
    
    @property
    def total(self) -> int:
        return self.nodes[0]
    
    def update(self, idx: int, value: _Number):
        idx = idx + self.capacity - 1  # child index in a tree arr
        change = value - self.nodes[idx]

        self.nodes[idx] = value
        
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2
    
    def add(self, value: _Number, data: int):
        self.data[self.pointer] = data
        self.update(self.pointer, value)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def get(self, cumsum: _Number):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            
            left = 2 * idx + 1
            right = left + 1
            
            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]
        
        idx = idx - self.capacity + 1
        return idx, self.nodes[idx], self.data[idx]

    def __repr__(self) -> str:
        return f"HeapSum(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


class PrioritizedReplayBuffer:
    def __init__(self,
                 buffer: ReplayBuffer,
                 eps: float = 1e-2,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 device: Union[str, torch.device] = "cpu") -> None:
        self.tree = HeapSum(capacity=buffer.buffer_size)

        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines the amount of "non-uniformness" and prioritization impact
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples
        self.device = device

        # i/o order: state, action, reward, next_state, done
        self.state = buffer.states
        self.action = buffer.actions
        self.reward = buffer.rewards
        self.next_state = buffer.next_states
        self.done = buffer.dones

        self.pointer = 0
        self.size = 0
        self.capacity = buffer.buffer_size
    
    def to_tensor(self, data: Union[np.ndarray, List[Union[float, int]]]) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float, device=self.device)
    
    def add(self, transition: List[torch.Tensor]):
        state, action, reward, next_state, done = transition

        self.tree.add(self.max_priority, self.pointer)

        self.state[self.pointer] = self.to_tensor(state)
        self.action[self.pointer] = self.to_tensor(action)
        self.reward[self.pointer] = self.to_tensor(reward)
        self.next_state[self.pointer] = self.to_tensor(next_state)
        self.done[self.pointer] = self.to_tensor(done)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
    
    def sample(self, batch_size: int) -> Tuple[List[torch.Tensor], torch.Tensor, list]:
        assert self.size >= batch_size

        sample_indexes, tree_indexes = [], []
        priorities = torch.empty(batch_size, 1, device=self.device, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = a + segment

            cumsum = uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_indexes.append(tree_idx)
            sample_indexes.append(sample_idx)
        
        probs = priorities / self.tree.total

        weights = (self.size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = [
            self.state[sample_indexes],
            self.action[sample_indexes],
            self.reward[sample_indexes],
            self.next_state[sample_indexes],
            self.action[np.array(sample_indexes) + 1]
        ]

        return batch, weights, tree_indexes
    
    def update_priorities(self, indexes: Union[List[int], torch.Tensor, np.ndarray], priorities: Union[List[float], torch.Tensor, np.ndarray]):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        
        for idx, priority in zip(indexes, priorities):
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

import os
from typing import Union, Tuple, Dict
import torch
from torch import nn
import numpy as np
from tqdm import tqdm


eps = 1e-10


class OnlineReplayBuffer:
    def __init__(self,
                 device: str,
                 state_dim: int,
                 action_dim: int,
                 buffer_size: int) -> None:
        self.device = device
        self.size = 0
        self.pointer = 0
        self.buffer_size = buffer_size

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.next_actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    @staticmethod
    def to_tensor(data: np.ndarray, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.tensor(data, dtype=torch.float32, device=device)

    def add_transition(self,
                       state: Union[torch.Tensor, np.ndarray],
                       action: Union[torch.Tensor, np.ndarray],
                       reward: Union[torch.Tensor, np.ndarray],
                       next_state: Union[torch.Tensor, np.ndarray],
                       next_action: Union[torch.Tensor, np.ndarray],
                       done: Union[torch.Tensor, np.ndarray]):
        if not isinstance(state, torch.Tensor):
            state = self.to_tensor(state, self.device)
            action = self.to_tensor(action, self.device)
            reward = self.to_tensor(reward, self.device)
            next_state = self.to_tensor(next_state, self.device)
            next_action = self.to_tensor(next_action, self.device)
            done = self.to_tensor(done, self.device)

        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.next_actions[self.pointer] = next_action
        self.dones[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_return(self, gamma: float) -> None:
        _return = 0
        
        for i in tqdm(reversed(range(self.size)), desc="Computing returns"):
            self.returns[i] = self.rewards[i] + gamma * _return * (1 - self.dones[i])
            _return = self.returns[i]
    
    def compute_advantage(self,
                          gamma: float,
                          lmbda: float,
                          value_fn: nn.Module) -> None:
        delta = torch.zeros_like(self.rewards)
        v = value_fn.to(self.device)

        _value = 0
        _advantage = 0

        for i in tqdm(reversed(range(self.size)), desc="Computing advantages"):
            current_value = v(self.states[i]).flatten()

            delta[i] = self.rewards[i] + gamma * _value * (1.0 - self.dones[i]) - current_value
            self.advantages[i] = delta[i] + gamma * lmbda * _advantage * (1.0 - self.dones[i])

            _value = current_value
            _advantage = self.advantages[i]
        
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + eps)

    def sample(self, batch_size: int):
        indexes = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.next_states[indexes],
            self.next_actions[indexes],
            self.dones[indexes],
            self.returns[indexes],
            self.advantages[indexes]
        )


class OfflineReplayBuffer(OnlineReplayBuffer):
    def __init__(self, device: str, state_dim: int, action_dim: int, buffer_size: int) -> None:
        super().__init__(device, state_dim, action_dim, buffer_size)
    
    def from_d4rl(self, dataset: Dict[str, np.ndarray]) -> None:
        if self.size:
            print("Warning: loading data into non-empty buffer")
        n_transitions = dataset["observations"].shape[0]

        if n_transitions < self.buffer_size:
            self.states[:n_transitions] = self.to_tensor(dataset["observations"][-n_transitions:], self.device)
            self.actions[:n_transitions] = self.to_tensor(dataset["actions"][-n_transitions:], self.device)
            self.next_states[:n_transitions] = self.to_tensor(dataset["next_observations"][-n_transitions:], self.device)
            self.rewards[:n_transitions] = self.to_tensor(dataset["rewards"][-n_transitions:].reshape(-1, 1), self.device)
            self.dones[:n_transitions] = self.to_tensor(dataset["terminals"][-n_transitions:].reshape(-1, 1), self.device)

        else:
            self.buffer_size = n_transitions

            self.states = self.to_tensor(dataset["observations"][-n_transitions:], self.device)
            self.actions = self.to_tensor(dataset["actions"][-n_transitions:-1])
            self.next_states = self.to_tensor(dataset["next_observations"][-n_transitions:], self.device)
            self.next_actions = self.to_tensor(dataset["actions"][-n_transitions + 1:], self.device)
            self.rewards = self.to_tensor(dataset["rewards"][-n_transitions:].reshape(-1, 1), self.device)
            self.dones = self.to_tensor(dataset["terminals"][-n_transitions:].reshape(-1, 1), self.device)
        
        self.size = n_transitions
        self.pointer = n_transitions % self.buffer_size

    def from_json(self, json_file: str):
        import json

        if not json_file.endswith('.json'):
            json_file = json_file + '.json'

        json_file = os.path.join("json_datasets", json_file)
        output = dict()

        with open(json_file) as f:
            dataset = json.load(f)
        
        for k, v in dataset.items():
            v = np.array(v)
            if k != "terminals":
                v = v.astype(np.float32)
            
            output[k] = v
        
        self.from_d4rl(output)

    def normalize_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.states.mean(0, keepdim=True)
        std = self.states.std(0, keepdim=True) + eps
        self.states = (self.states - mean) / std
        self.next_states = (self.next_states - mean) / std
        return mean, std

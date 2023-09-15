import os
from typing import List, Dict, Tuple
import numpy as np
import torch


class LAP:
    '''
        Loss-Adjusted Prioritized Experience Replay

        https://arxiv.org/abs/2007.06049
    '''
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 device: str = "cpu",
                 buffer_size: int = 1_000_000,
                 max_action: float = 1.0,
                 normalize_actions: bool = True,
                 with_priority: bool = True) -> None:
        
        self.buffer_size = buffer_size
        self.device = device

        self.pointer = 0
        self.size = 0

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self.with_priortiy = with_priority

        if with_priority:
            self.priortiy = torch.zeros(buffer_size, device=device)
            self.max_priority = 1.0
        
        self.normalizing_factor = max_action if normalize_actions else 1.0
    
    @staticmethod
    def to_tensor(data: np.ndarray, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.tensor(data, dtype=torch.float32, device=device)
    
    def add_transition(self,
                       state: torch.Tensor,
                       action: torch.Tensor,
                       reward: torch.Tensor,
                       next_state: torch.Tensor,
                       done: torch.Tensor):
        if not isinstance(state, torch.Tensor):
            state = self.to_tensor(state, self.device)
            action = self.to_tensor(action, self.device)
            reward = self.to_tensor(reward, self.device)
            next_state = self.to_tensor(next_state, self.device)
            done = self.to_tensor(done, self.device)


        self.states[self.pointer] = state
        self.actions[self.pointer] = action / self.normalizing_factor
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done

        if self.with_priortiy:
            self.priortiy[self.pointer] = self.max_priority

        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def add_batch(self,
                  states: List[torch.Tensor],
                  actions: List[torch.Tensor],
                  rewards: List[torch.Tensor],
                  next_states: List[torch.Tensor],
                  dones: List[torch.Tensor]):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.add_transition(state, action, reward, next_state, done)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.with_priortiy:
            cumsum = torch.cumsum(self.priortiy[:self.size], dim=0)
            value = torch.rand(size=(batch_size,), device=self.device) * cumsum[-1]
            self.indexes: np.ndarray = torch.searchsorted(cumsum, value).cpu().data.numpy()
        else:
            self.indexes = np.random.randint(0, self.size, size=batch_size)
        
        # states actions rewards next_states dones
        return (
            self.states[self.indexes],
            self.actions[self.indexes],
            self.rewards[self.indexes],
            self.next_states[self.indexes],
            self.dones[self.indexes]
        )

    def update_priority(self, priority: torch.Tensor):
        self.priortiy[self.indexes] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def update_max_priority(self):
        self.max_priority = float(self.priortiy[:self.size].max())

    def from_d4rl(self, dataset: Dict[str, np.ndarray]):
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
            self.actions = self.to_tensor(dataset["actions"][-n_transitions:])
            self.next_states = self.to_tensor(dataset["next_observations"][-n_transitions:], self.device)
            self.rewards = self.to_tensor(dataset["rewards"][-n_transitions:].reshape(-1, 1), self.device)
            self.dones = self.to_tensor(dataset["terminals"][-n_transitions:].reshape(-1, 1), self.device)
        
        self.size = n_transitions
        self.pointer = n_transitions % self.buffer_size

        if self.with_priortiy:
            self.priortiy = torch.ones(self.size).to(self.device)

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
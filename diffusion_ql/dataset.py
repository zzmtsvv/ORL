import torch
import numpy as np
from typing import List, Tuple
import os


class ReplayBuffer:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 buffer_size: int = 1000000) -> None:
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.pointer = 0
        self.size = 0

        device = "cpu"
        self.device = device

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        # i/o order: state, action, reward, next_state, done
    
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
    
    def get_moments(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        state_mean, state_std = self.states.mean(dim=0), self.states.std(dim=0)
        action_mean, action_std = self.actions.mean(dim=0), self.actions.std(dim=0)

        return (state_mean, state_std), (action_mean, action_std)
    
    @staticmethod
    def to_tensor(data: np.ndarray, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.tensor(data, dtype=torch.float32, device=device)
    
    def sample(self, batch_size: int):
        indexes = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.next_states[indexes],
            self.dones[indexes]
        )
    
    def from_d4rl(self, dataset):
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
    
    def from_d4rl_finetune(self, dataset):
        raise NotImplementedError()
    
    def normalize_states(self, eps=1e-3):
        mean = self.states.mean(0, keepdim=True)
        std = self.states.std(0, keepdim=True) + eps
        self.states = (self.states - mean) / std
        self.next_states = (self.next_states - mean) / std
        return mean, std
    
    def clip(self, eps=1e-5):
        self.actions = torch.clip(self.actions, - 1 + eps, 1 - eps)

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
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done

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
    
    @staticmethod
    def dataset_stats(dataset):
        episode_returns = []
        returns = 0
        episode_length = 0

        for reward, done in zip(dataset["rewards"], dataset["terminals"]):
            if done:
                episode_returns.append(returns)
                returns = 0
                episode_length = 0
            else:
                episode_length += 1
                returns += reward
                if episode_length == 1000:
                    episode_returns.append(returns)
                    returns = 0
                    episode_length = 0

        episode_returns = np.array(episode_returns)
        return episode_returns.mean(), episode_returns.std()
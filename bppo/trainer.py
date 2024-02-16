from dataclasses import asdict
from config import bppo_config
from buffer import OfflineReplayBuffer
from bppo import BPPO

import wandb
from tqdm import tqdm


class BPPOTrainer:
    def __init__(self,
                 cfg=bppo_config) -> None:
        self.cfg = cfg

        self.replay_buffer = OfflineReplayBuffer(cfg.device, cfg.state_dim, cfg.action_dim, cfg.buffer_size)
        self.replay_buffer.from_json(cfg.dataset_name)
        self.replay_buffer.compute_return(cfg.gamma)
        self.replay_buffer.normalize_states()

        self.bppo = BPPO(cfg)
    
        wandb.init(
            project=self.cfg.project,
            group=self.cfg.group,
            name=self.cfg.name,
            config=asdict(self.cfg)
        )
    
    def train(self):
        print(f"Training starts on {self.cfg.device}🚀")

        for step in tqdm(range(self.cfg.value_steps), desc="Value Func Training"):
            states, _, _, _, _, _, returns, _ = self.replay_buffer.sample(self.cfg.batch_size)
            value_loss = self.bppo.value_update(states, returns)

        for step in tqdm(range(self.cfg.critic_steps), desc="Critic Training"):
            states, actions, rewards, next_states, next_actions, dones, _, _ = self.replay_buffer.sample(self.cfg.batch_size)
            critic_loss = self.bppo.critic_update(states, actions, rewards, next_states, next_actions, dones)

            if not (step + 1) % self.cfg.target_update_freq:
                self.bppo.soft_critic_update()

        for step in tqdm(range(self.cfg.bc_steps), desc="Behavior Cloning Pretraining"):
            states, actions, _, _, _, _, _, _ = self.replay_buffer.sample(self.cfg.batch_size)
            bc_loss = self.bppo.behavior_cloning_update(states, actions)

        for step in tqdm(range(self.cfg.bppo_steps), desc="BPPO Training"):
            # as in official implementation & as stated in the paper
            if step > 200:
                self.bppo.clip_decay = False
                self.bppo.lr_decay = False
            
            states, _, _, _, _, _, _, _ = self.replay_buffer.sample(self.cfg.batch_size)
            policy_loss = self.bppo.policy_update(states)


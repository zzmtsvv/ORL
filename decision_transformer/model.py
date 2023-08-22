from typing import Optional
import torch
from torch import nn
from config import dt_config
from block import TransformerBlock


class DecisionTransformer(nn.Module):
    def __init__(self,
                 cfg: dt_config,
                 state_dim: int,
                 action_dim: int) -> None:
        super().__init__()

        self.embedding_dropout = nn.Dropout(cfg.embedding_dropout)
        self.embedding_norm = nn.LayerNorm(cfg.embedding_dim)

        self.final_norm = nn.LayerNorm(cfg.embedding_dim)
        self.positional_encoding = nn.Embedding(cfg.episode_length + cfg.sequence_length,
                                           cfg.embedding_dim)
        self.state_embedding = nn.Linear(state_dim, cfg.embedding_dim)
        self.action_embedding = nn.Linear(action_dim, cfg.embedding_dim)
        self.return_embedding = nn.Linear(1, cfg.embedding_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(3 * cfg.sequence_length,
                             cfg.embedding_dim,
                             cfg.num_heads,
                             cfg.attention_dropout,
                             cfg.residual_dropout) for _ in range(cfg.num_layers)
        ])
        
        self.embedding_dim = cfg.embedding_dim
        self.sequence_length = cfg.sequence_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = cfg.episode_length
        self.max_action = cfg.max_action

        self.action_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.action_dim),
            nn.Tanh()
        )

        self.apply(self.reset_weights)

    @staticmethod
    def reset_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor,
                mc_returns: torch.Tensor,
                time_steps: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, sequence_length = states.shape[0], states.shape[1]

        pos_encoding = self.positional_encoding(time_steps)
        state_embedding = self.state_embedding(states) + pos_encoding
        action_embedding = self.action_embedding(actions) + pos_encoding
        returns_embedding = self.return_embedding(mc_returns.unsqueeze(-1)) + pos_encoding

        sequence = torch.stack((
            returns_embedding, state_embedding, action_embedding
        ), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * sequence_length, self.embedding_dim)

        if key_padding_mask is not None:
            key_padding_mask = torch.stack((
                key_padding_mask, key_padding_mask, key_padding_mask
            ), dim=1).permute(0, 2, 1).reshape(batch_size, 3 * sequence_length)
        
        out = self.embedding_dropout(self.embedding_norm(sequence))

        for block in self.blocks:
            out = block(out, padding_mask=key_padding_mask)
        
        out = self.final_norm(out)
        return self.action_head(out[:, 1::3]) * self.max_action

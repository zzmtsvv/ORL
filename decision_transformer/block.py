from typing import Optional
import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self,
                 sequence_length: int,
                 embedding_dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 residual_dropout: float) -> None:
        super().__init__()

        self.sequence_length = sequence_length

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(residual_dropout)

        self.self_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True)
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout)
        )

        self.causal_mask = ~torch.tril(torch.ones(sequence_length, sequence_length)).to(bool)
    
    def forward(self,
                x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]]
        causal_mask = causal_mask.to(x.device)

        x_prime = self.layer_norm1(x)
        self_attention = self.self_attention(query=x_prime,
                                        key=x_prime,
                                        value=x_prime,
                                        attn_mask=causal_mask,
                                        key_padding_mask=key_padding_mask,
                                        need_weights=False)[0]
        x = x + self.dropout(self_attention)
        x = x + self.net(self.layer_norm2(x))
        return x


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True):
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias)
        
        padding_size = (kernel_size - 1) * dilation
        self.zero_padding = nn.ConstantPad1d(padding=(padding_size, 0), value=0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded_input = self.zero_padding(x)
        return super().forward(padded_input)

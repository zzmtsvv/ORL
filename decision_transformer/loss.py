import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


class DTLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self,
                predicted_actions: torch.Tensor,
                actions: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")

        loss = loss * mask.unsqueeze(-1)
        return loss.mean()

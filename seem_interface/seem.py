from math import sqrt
from typing import Union, List, Dict
import torch
from torch import nn


OfflineRLBatchType = List[torch.Tensor]


class SEEM:
    '''
        This class is considered as an example with important details to be considered
        while integrating this idea into your class method. This class does not work
        and is not supposed to do so. It is just the manual where to put some things
        to implement SEEM for your own algorihm.

        Hope this would be useful for you :)
    '''
    def __init__(self,
                 device: str,
                 discount: float,  # discount factor, a.k.a gamma
                 ntk_states: torch.Tensor,
                 ntk_actions: torch.Tensor,
                 ntk_next_states: torch.Tensor) -> None:
        '''
        ntk tensors are fixed inputs from offline dataset for fair tracking. they should be of shape
        [N, dim] - where N can be less than batch_size, dim is dimension of state or action
        features respectively
        '''
        self.device = device
        self.ntk_states = ntk_states.to(device)
        self.ntk_actions = ntk_actions.to(device)
        self.ntk_next_states = ntk_next_states.to(device)

        # __init__ for your method
        self.discount = discount
        self.actor: nn.Module
        self.critic1: nn.Module
        self.critic2: nn.Module
        self.critic_optim: torch.optim.Optimizer
        self.actor_optim: torch.optim.Optimizer
        ...

    @staticmethod
    def flatten_gradients(gradients: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        return torch.cat([g.view(-1) for g in gradients])
    
    @staticmethod
    def model_weights_norm(model: nn.Module) -> float:
        flattenned_parameters = torch.cat([p.view(-1) for p in model.parameters()])
        return torch.norm(flattenned_parameters, p=2).item()
    
    @staticmethod
    def layer_norm(model: nn.Module, layer_index: int) -> float:
        # used for more detailed tracking on what's going on inside actor and critic networks
        # assert layer_index in range(4), f"wrong layer index {layer_index} for {model.__class__.__name__}"

        layer = [layer for layer in model if "Linear" in layer.__class__.__name__][layer_index]
        layer_parameters = torch.cat([p.view(-1) for p in layer.parameters()])
        return torch.norm(layer_parameters, p=2).item()

    def seem_statistics(self) -> Dict[str, float]:
        grads = []
        for i in range(self.ntk_states.size(0)):
            # take q1 according to official realization
            output: torch.Tensor = self.critic1(self.ntk_states[i].unsqueeze(0), self.ntk_actions[i].unsqueeze(0))
            self.critic1.zero_grad()
            output.backward()

            grad = self.flatten_gradients([p.grad for p in self.critic1.parameters() if p.grad is not None])
            grads.append(grad)
        
        grads = (torch.stack(grads) / sqrt(grad.shape[0])).detach()
        # G = torch.matmul(grads, grads.T)

        with torch.no_grad():
            # keep attention to this line, it quiet depends on actor `forward` method
            next_pi = self.actor(self.ntk_next_states)

        next_grads = []
        for i in range(self.ntk_next_states.size(0)):
            output = self.critic1(self.ntk_next_states[i].unsqueeze(0), next_pi[i].unsqueeze(0))
            self.critic1.zero_grad()
            output.backward()

            grad = self.flatten_gradients([p.grad for p in self.critic1.parameters() if p.grad is not None])
            next_grads.append(grad)
        
        next_pi_grads = (torch.stack(next_grads) / sqrt(grad.shape[0])).detach()

        A = self.discount * torch.matmul(next_pi_grads, grads.T) - torch.matmul(grads, grads.T)
        eigenvalues = torch.linalg.eigvals(A).real
        normed_eigenvalues = eigenvalues / torch.pow(A, 2).sum().sqrt()
        
        return {
            "seem/max_eigenvalue": eigenvalues.max().cpu().item(),
            "seem/max_normed_eigenvalue": normed_eigenvalues.max().cpu().item(),
            "seem/critic1_grad_norm": grads.norm(p=2).item()
        }
    
    def train_step(self, data: OfflineRLBatchType) -> Dict[str, float]:
        
        critic_loss = ...
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        ...

        actor_loss = ...

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        ...

        seem_info = self.seem_statistics()

        # returns seem statistics together with the statistics of your main algorithm
        return {
            **seem_info,
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }

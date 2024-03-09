from copy import deepcopy
from typing import Union, Dict
import torch
from modules import ValueFunction, DeterministicActor, StochasticActor


def reverse_kl(residual: torch.Tensor) -> torch.Tensor:
    return (residual - 1).exp()


def pearson_chi_square(residual: torch.Tensor) -> torch.Tensor:
    w = (residual / 2 + 1).clamp_min(0.0)
    return residual * w - (w - 1).pow(2)


def inverse_pearson(residual: torch.Tensor) -> torch.Tensor:
    return residual.clamp_min(0.0)


class ODICE:
    def __init__(self,
                 device: str,
                 value_func: ValueFunction,
                 actor: Union[DeterministicActor, StochasticActor],
                 exp_adv_max: float = 100.0,
                 convex_conj_func: str = "reverse_kl",
                 lmbda: float = 0.8,
                 value_func_lr: float = 3e-4,
                 actor_lr: float = 3e-4,
                 weight_decay: float = 1e-5,
                 eta: float = 1.0,
                 discount: float = 0.99,
                 tau: float = 5e-3) -> None:
        self.device = device

        self.exp_adv_max = exp_adv_max

        self.value_func = value_func.to(device)
        self.value_optim = torch.optim.Adam(self.value_func.parameters(), lr=value_func_lr)

        with torch.no_grad():
            self.value_target = deepcopy(self.value_func)

        self.actor = actor.to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)

        assert convex_conj_func in ("reverse_kl", "pearson_chi_square")

        if convex_conj_func == "reverse_kl":
            self.f_prime = reverse_kl
            self.f_inverse = reverse_kl
        if convex_conj_func == "pearson_chi_square":
            self.f_prime = pearson_chi_square
            self.f_inverse = inverse_pearson

        self.lmdbda = lmbda
        self.eta = eta
        self.discount = discount
        self.tau = tau

        self.total_iterations = 0

    def soft_value_update(self):
        for param, tgt_param in zip(self.value_func.parameters(), self.value_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
    
    def ortho_true_grad_update(self,
                               states: torch.Tensor,
                               actions: torch.Tensor,
                               rewards: torch.Tensor,
                               next_states: torch.Tensor,
                               dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        with torch.no_grad():
            tgt_value = self.value_target(states)
            tgt_value_next = self.value_target(next_states)
        
        value = self.value_func(states)
        value_next = self.value_func(next_states)
        
        forward_td_error = rewards + (1.0 - dones) * self.discount * tgt_value_next - value
        backward_td_error = rewards + (1.0 - dones) * self.discount * value_next - tgt_value

        forward_dual_loss = (self.lmdbda * self.f_prime(forward_td_error)).mean()
        backward_dual_loss = (self.lmdbda * self.eta * self.f_prime(backward_td_error)).mean()

        actor_residual = forward_td_error.clone().detach()

        self.value_optim.zero_grad(set_to_none=True)
        fwd_grads, bckwrd_grads = [], []

        forward_dual_loss.backward(retain_graph=True)
        fwd_grads = [param.grad.clone().detach().reshape(-1) for param in list(self.value_func.parameters())]

        backward_dual_loss.backward()
        for i, param in enumerate(list(self.value_func.parameters())):
            bckwrd_grads.append(param.grad.clone().detach().reshape(-1) - fwd_grads[i])
        
        forward_grad, backward_grad = torch.cat(fwd_grads), torch.cat(bckwrd_grads)
        # gram-schmidt
        coef = (torch.dot(forward_grad, backward_grad) / (torch.dot(forward_grad, forward_grad) + 1e-10)).item()
        forward_grad = (1 - coef) * forward_grad + backward_grad

        idx = 0
        for i, grad in enumerate(fwd_grads):
            fwd_grads[i] = forward_grad[idx:idx + grad.shape[0]]
            idx += grad.shape[0]
        
        self.value_optim.zero_grad(set_to_none=True)
        ((1.0 - self.lmdbda) * value).mean().backward()
        
        for i, param in enumerate(list(self.value_func.parameters())):
            param.grad += fwd_grads[i].reshape(param.grad.shape)
        
        self.value_optim.step()

        self.soft_value_update()

        weight = self.f_inverse(actor_residual).clamp_max(self.exp_adv_max).detach()
        bc_loss = -self.actor.log_prob(states, actions)
        actor_loss = (weight * bc_loss).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "value": value.mean().item(),
            "td_mean": forward_td_error.mean().item(),
            "td_min": forward_td_error.min().item(),
            "td_max": forward_td_error.max().item(),
            "weight_min": weight.min().item(),
            "weight_max": weight.max().item(),
        }

    # def true_grad_update(self,
    #                      states: torch.Tensor,
    #                      actions: torch.Tensor,
    #                      rewards: torch.Tensor,
    #                      next_states: torch.Tensor,
    #                      dones: torch.Tensor) -> Dict[str, float]:
    #     self.total_iterations += 1

    #     target_value = self.value_func(next_states)
        
    #     value = self.value_func(states)
    #     td_error: torch.Tensor = rewards + (1.0 - dones) * self.discount * target_value - value
    #     dual_loss = self.f_prime(td_error)

    #     actor_residual = td_error.clone().detach()

    #     value_loss = ((1.0 - self.lmdbda) * value + self.lmdbda * dual_loss).mean()
    #     self.value_optim.zero_grad(set_to_none=True)
    #     value_loss.backward()
    #     self.value_optim.step()

    #     self.soft_value_update()

    #     weight = self.f_inverse(actor_residual).clamp_max(self.exp_adv_max).detach()
    #     bc_loss = -self.actor.log_prob(states, actions)
    #     actor_loss = (weight * bc_loss).mean()

    #     self.actor_optim.zero_grad(set_to_none=True)
    #     actor_loss.backward()
    #     self.actor_optim.step()

    #     return {
    #         "value": value.mean().item(),
    #         "td_mean": td_error.mean().item(),
    #         "td_min": td_error.min().item(),
    #         "td_max": td_error.max().item(),
    #         "weight_max": weight.max().item(),
    #         "weight_min": weight.min().item()
    #     }

    # def semi_grad_update(self,
    #                      states: torch.Tensor,
    #                      actions: torch.Tensor,
    #                      rewards: torch.Tensor,
    #                      next_states: torch.Tensor,
    #                      dones: torch.Tensor) -> Dict[str, float]:
    #     self.total_iterations += 1

    #     with torch.no_grad():
    #         target_value = self.value_target(next_states)
        
    #     value = self.value_func(states)
    #     td_error: torch.Tensor = rewards + (1.0 - dones) * self.discount * target_value - value
    #     dual_loss = self.f_prime(td_error)

    #     actor_residual = td_error.clone().detach()

    #     value_loss = ((1.0 - self.lmdbda) * value + self.lmdbda * dual_loss).mean()
    #     self.value_optim.zero_grad(set_to_none=True)
    #     value_loss.backward()
    #     self.value_optim.step()

    #     self.soft_value_update()

    #     weight = self.f_inverse(actor_residual).clamp_max(self.exp_adv_max).detach()
    #     bc_loss = -self.actor.log_prob(states, actions)
    #     actor_loss = (weight * bc_loss).mean()

    #     self.actor_optim.zero_grad(set_to_none=True)
    #     actor_loss.backward()
    #     self.actor_optim.step()

    #     return {
    #         "value": value.mean().item(),
    #         "td_mean": td_error.mean().item(),
    #         "td_min": td_error.min().item(),
    #         "td_max": td_error.max().item(),
    #         "weight_max": weight.max().item(),
    #         "weight_min": weight.min().item()
    #     }

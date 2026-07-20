import torch
from typing import Tuple


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_on_dim = torch.max(x, dim=dim, keepdim=True).values #torch.max return value type is subtle
    return torch.exp(x - max_on_dim) / torch.sum(torch.exp(x - max_on_dim), dim = dim, keepdim=True)

def log_sum_exp(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_on_dim = torch.max(x, dim = -1, keepdim=True).values
    return torch.log(torch.sum(torch.exp(x - max_on_dim), dim = -1, keepdim=True)) + max_on_dim # (batch_size, seq_len, 1)

def cross_entropy(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    lse = log_sum_exp(x) # (b, seq_len, 1)
    o_select = x.gather(dim = -1, index = targets.unsqueeze(-1)).unsqueeze(-1) 
    return (lse - o_select).mean()

def gradient_clipping(parameters, max_l2_norm, eps = 1e-6):
    parameter_group = [
        p for p in parameters if p.grad is not None
    ]
    if not parameter_group:
        return 
    norm = 0
    for p in parameter_group:
        norm += torch.norm(p.grad.data)**2
    norm = torch.sqrt(norm)
    if norm > max_l2_norm:
        for p in parameter_group:
            p.grad.data *= (max_l2_norm / (norm + eps))

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

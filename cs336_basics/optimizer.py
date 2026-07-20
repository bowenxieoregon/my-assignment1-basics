from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            lam = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data, device = p.data.device))
                v = state.get("v", torch.zeros_like(p.data, device = p.data.device))
                grad = p.grad.data 

                alpha_t = alpha * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha * lam * p.data
                state["m"] = beta1 * m + (1 - beta1) * grad
                state["v"] = beta2 * v + (1 - beta2) * (grad**2)

                p.data -= alpha_t * (state["m"] / (torch.sqrt(state["v"]) + eps))

                state["t"] = t + 1
        return loss

def learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it >= warmup_iters and it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

                


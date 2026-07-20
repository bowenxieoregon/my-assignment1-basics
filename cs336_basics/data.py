import torch
import numpy as np
from itertools import islice

def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: torch.device | None = None):
    max_start = len(x) - context_length
    x_t = torch.tensor(x, device = device)
    start_indices = torch.randint(low=0, high=max_start, size = (batch_size,), device = device) #(batch_size,)
    offsets = torch.arange(context_length, device = device) #(context_len,)

    all = start_indices.unsqueeze(1) + offsets.unsqueeze(0)

    return x_t[all], x_t[all + 1]
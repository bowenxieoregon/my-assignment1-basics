import torch
import os
import typing

def save_checkpointing(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str| os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    obj = {
        "model": model_state,
        "optimizer": optimizer_state,
        "iteration": iteration
    }
    torch.save(obj, out)

def load_checkpointing(src: str| os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    states = torch.load(src)
    model.load_state_dict(states["model"])
    optimizer.load_state_dict(states["optimizer"])

    return states["iteration"]
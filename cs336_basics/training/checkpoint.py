import os
import torch
import torch.nn as nn
import torch.optim as optim
import typing
def save_checkpoint(
    model : nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    ckpts = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(ckpts, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    ckpts = torch.load(src)
    model.load_state_dict(ckpts["model"])
    optimizer.load_state_dict(ckpts["optimizer"])
    return ckpts["iteration"]
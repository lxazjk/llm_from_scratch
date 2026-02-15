import os
import tqdm
import torch
import typing
import numpy as np
import torch.nn as nn
import torch.optim as optim
from cs336_basics.training.checkpoint import save_checkpoint
def train(
    model: nn.Module,
    optim: optim.Optimizer,
    num_step: int,
    dataset_dir: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    batch_size: int,
    context_length: int,
    device: str,
    data_loader: typing.Callable[
        [np.array, int, int, str], typing.Tuple[torch.tensor, torch.tensor]
    ],
    Loss : typing.Callable[
        [torch.tensor, torch.tensor], torch.tensor
    ],
):
    dataset = np.memmap(dataset_dir, dtype="int32", mode="r")
    model.train()
    pbar = tqdm.tqdm(range(num_step))
    for _ in pbar:
        optim.zero_grad()
        inputs, target = data_loader(dataset, batch_size, context_length, device)
        logits = model(inputs)
        loss = Loss(logits, target).mean()
        loss.backward()
        optim.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if (_ + 1) % 1000 == 0:
            save_checkpoint(
                model = model,
                optimizer = optim,
                step = _
            )

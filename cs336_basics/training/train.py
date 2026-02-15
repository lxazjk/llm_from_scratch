import os
import tqdm
import torch
import typing
import numpy as np
import torch.nn as nn
import torch.optim as optim
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
    dataloader = data_loader(
        dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )
    model.train()
    for _ in tqdm.tqdm(num_step):
        optim.zero_grad()
        inputs, target = dataloader()
        logits = model(inputs)
        loss = Loss(logits, target)
        loss.backward()
        optim.step()
import os
import tqdm
import typing
import numpy as np
import torch.nn as nn
import torch.optim as optim
from cs336_basics.training.dataloader import data_loader
from cs336_basics.optimizer.Loss import Cross_Entropy
def train(
    model: nn.Module,
    optim: optim.Optimizer,
    num_step: int,
    dataset_dir: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    batch_size: int,
    context_length: int,
    device: str
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
        loss = Cross_Entropy(logits, target)
        loss.backward()
        optim.step()
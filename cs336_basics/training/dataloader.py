import numpy as np
import torch

def data_loader(
    dataset: np.array,
    batch_size: int,
    context_length: int,
    device
):
    dataset_length = dataset.shape[0]
    start_indices = np.random.randint(0, dataset_length - context_length, size = batch_size)
    input_ids = torch.stack([torch.from_numpy(dataset[i:i+context_length]) for i in start_indices], dim = 0).to(device)
    target_ids = torch.stack([torch.from_numpy(dataset[i+1:i+context_length+1]) for i in start_indices], dim = 0).to(device)
    return input_ids, target_ids

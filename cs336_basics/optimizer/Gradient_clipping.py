import torch
import math

import torch
from typing import Iterable

def Gradient_Clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon = 1e-6):
    total_norm_sq = 0.0
    epsilon = 1e-6
    params = list(parameters)
    for p in params:
        if p.grad is not None:
            total_norm_sq += torch.sum(p.grad.detach() ** 2)
    total_norm = torch.sqrt(total_norm_sq)
    if total_norm > max_l2_norm:
        coeff = max_l2_norm / (total_norm + epsilon)
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(coeff)
import torch
import torch.nn as nn

def SiLU(
    input: torch.tensor
) -> torch.tensor:
    in_dtype = input.dtype
    input = input.to(in_dtype)
    return (input / (torch.exp(-input) + 1)).to(in_dtype)
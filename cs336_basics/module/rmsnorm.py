import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(
            torch.empty(
                d_model,
                device = self.device,
                dtype = self.dtype
            )
        )
        self._init_weight()
    
    def _init_weight(self):
        nn.init.trunc_normal_(self.weight, mean = 1) # ?
    
    def forward(
        self,
        x : torch.tensor
    ) -> torch.tensor :
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_sqr = x * x
        RMS_x = torch.sqrt(x_sqr.mean(dim = -1, keepdim=True) + self.eps)
        return (x / RMS_x * self.weight).to(in_dtype)
import torch
import torch.nn as nn
from cs336_basics.module.Linear import Linear
from cs336_basics.module.SiLu import SiLU

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        self.SiLU = SiLU
        self.Linear1 = Linear(
            in_feature = self.d_model,
            out_feature = self.d_ff,
            device = self.device,
            dtype = self.dtype
        )
        self.Linear2 = Linear(
            in_feature = self.d_ff,
            out_feature = self.d_model,
            device = self.device,
            dtype = self.dtype
        )
        self.Linear3 = Linear(
            in_feature = self.d_model,
            out_feature = self.d_ff,
            device = self.device,
            dtype = self.dtype
        )
    
    def forward(
        self,
        input: torch.tensor
    ) -> torch.tensor:
        return self.Linear2(self.SiLU(self.Linear1(input)) * self.Linear3(input))
    
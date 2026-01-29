import torch
import torch.nn as nn
from cs336_basics.module.MultiheadSelfAttention import MultiheadSelfAttention
from cs336_basics.module.RMSnorm import RMSNorm
from cs336_basics.module.SwiGLU import SwiGLU
class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32 
        self.RMSNorm_layer_1 = RMSNorm(
            d_model = d_model,
            device = device,
            dtype = dtype
        )
        self.RMSNorm_layer_2 = RMSNorm(
            d_model = d_model,
            device = device,
            dtype = dtype
        )
        self.MHA_layer = MultiheadSelfAttention(
            d_model = self.d_model,
            num_heads = self.num_heads,
            theta = self.theta,
            apply_rope = True,
            max_seq_len = max_seq_len,
            device = self.device,
            dtype = self.dtype
        )
        self.SwiGLU_layer = SwiGLU(
            d_ff = self.d_ff,
            d_model = d_model,
            device = self.device,
            dtype = self.dtype
        )
        
    def forward(
        self,
        input: torch.tensor
    ) -> torch.tensor:
        output_block = input + self.MHA_layer(self.RMSNorm_layer_1(input))
        return output_block + self.SwiGLU_layer(self.RMSNorm_layer_2(output_block))
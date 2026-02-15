import torch
import torch.nn as nn
from cs336_basics.module.Linear import Linear
from cs336_basics.module.Scaled_Dot_Product_Attention import Scaled_Dot_Product_Attention
from cs336_basics.module.Rope import Rope
from einops import rearrange
from typing import Optional
class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: Optional[float] = None,
        apply_rope: bool = False,
        max_seq_len: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) : 
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.apply_rope = apply_rope
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        assert (theta is not None and apply_rope) or (theta is None and apply_rope == False), "theta should (not) be provided when apply_rope is ture(false)"
        if self.apply_rope:
            self.Rope = Rope(
                dim = self.d_k,
                theta = self.theta,
                max_seq_len = self.max_seq_len,
                device = self.device,
                dtype = self.dtype
            )
        else:
            self.Rope = None
        position = torch.arange(0, self.max_seq_len, device = self.device, dtype = self.dtype)
        self.causal_mask = position.unsqueeze(0) <= position.unsqueeze(1)
        self.q_proj = Linear(
            self.num_heads * self.d_k,
            self.d_model,
            self.device,
            self.dtype
        )
        self.k_proj = Linear(
            self.num_heads * self.d_k,
            self.d_model,
            self.device,
            self.dtype
        )
        self.v_proj = Linear(
            self.num_heads * self.d_v,
            self.d_model,
            self.device,
            self.dtype
        )

        self.o_proj = Linear(
            self.d_model,
            self.num_heads * self.d_v,
            self.device,
            self.dtype
        )

    def forward(
        self,
        input: torch.tensor
    ) -> torch.tensor :
        batch_size, seq_len, d_model = input.shape
        assert d_model == self.d_model, f"input shape mismatch d_model {d_model}, desired {self.d_modelf}"
        # print(f"seq_length {seq_len} input.shape {input.shape}")
        # print(f"shape{self.q_proj(input).shape}")
        V = rearrange(self.v_proj(input), "b s (h d) -> b h s d", h = self.num_heads).contiguous()
        Q = rearrange(self.q_proj(input), "b s (h d) -> b h s d", h = self.num_heads).contiguous()
        K = rearrange(self.k_proj(input), "b s (h d) -> b h s d", h = self.num_heads).contiguous()
        shape = input.shape
        if self.apply_rope:
            Q = self.Rope(Q)
            K = self.Rope(K)
        # print(f"inputshape {shape} ------> ropeshape {input.shape}")
        # print(f"Q.shape {Q.shape} K.shape {K.shape} V.shape {V.shape}")
        mask = self.causal_mask[: seq_len, : seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)
        return self.o_proj(rearrange(Scaled_Dot_Product_Attention(Q, K, V, mask), "b h s d -> b s (h d)"))

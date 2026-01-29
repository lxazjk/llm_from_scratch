import torch
import torch.nn as nn

class Rope(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim{dim} should mod 2 = 0"
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        k = torch.arange(0, dim // 2, dtype = self.dtype, device = self.device)
        freq = self.theta ** (- 2.0 * k / dim)
        position = torch.arange(0, self.max_seq_len, device = self.device, dtype = self.dtype)
        angles = position.unsqueeze(-1) * freq.unsqueeze(0)
        self.register_buffer("sin_angle", torch.sin(angles))
        self.register_buffer("cos_angle", torch.cos(angles))
    
    def forward(
        self, 
        input: torch.tensor
    ) -> torch.tensor :
        seq_len = input.shape[-2]
        x_1 = input[..., 0::2]
        x_2 = input[..., 1::2]
        sin = self.sin_angle[:seq_len,]
        cos = self.cos_angle[:seq_len,]
        rot_x_1 = x_1 * cos - x_2 * sin
        rot_x_2 = x_1 * sin + x_2 * cos
        return torch.stack([rot_x_1, rot_x_2], dim = -1).flatten(-2)
        
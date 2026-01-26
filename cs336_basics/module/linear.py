import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(
        self, 
        in_feature, 
        out_feature, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.out_feature, self.in_feature),
                device = self.device, 
                dtype = self.dtype
            )
        )
        self._init_weight()
        
    def _init_weight(self):
        std_sqrt = math.sqrt(2.0 / (self.in_feature + self.out_feature))
        nn.init.trunc_normal_(self.weight, 
                              mean = 0.0, 
                              std = 2.0 / (self.in_feature + self.out_feature), 
                              a = -3.0 * std_sqrt, 
                              b = 3.0 * std_sqrt
                            )
    
    def forward(self, x):
        return torch.einsum("o i, ... i -> ... o", self.weight, x)
    
if __name__ == "__main__" :
    a = Linear(2, 2)
    print(f"a {a.weight} shape {a.weight.shape}")
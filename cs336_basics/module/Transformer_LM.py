import torch
import torch.nn as nn
from cs336_basics.module.Embedding import Embedding
from cs336_basics.module.Transformer import Transformer
from cs336_basics.module.RMSnorm import RMSNorm
from cs336_basics.module.Linear import Linear
from cs336_basics.module.Softmax import softmax
class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        self.Embedding_layer = Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.d_model,
            device = self.device,
            dtype = self.dtype
        )
        self.Transformer_layers = nn.ModuleList(
            [Transformer(
                d_model = self.d_model,
                d_ff = self.d_ff,
                num_heads = self.num_heads,
                max_seq_len = self.context_length,
                theta = self.rope_theta,
                device = self.device,
                dtype = self.dtype
            ) for _ in range(num_layers)]
        )
        self.RMSNorm_layer = RMSNorm(
            d_model = self.d_model,
            device = self.device,
            dtype = self.dtype
        )
        self.Linear_layer = Linear(
            in_feature = self.d_model,
            out_feature = self.vocab_size,
            device = self.device,
            dtype = self.dtype
        )
        self.softmax = softmax

    def forward(
        self,
        in_features: torch.tensor
    ) -> torch.tensor:
        output = self.Embedding_layer(in_features)
        for transformer_layer in self.Transformer_layers:
            output = transformer_layer(output)
        return self.Linear_layer(self.RMSNorm_layer(output))

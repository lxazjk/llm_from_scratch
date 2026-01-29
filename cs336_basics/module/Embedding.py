import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device if device is not None else "cpu"
        self.dtype = dtype if dtype is not None else torch.float32
        self.embedding = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device = device,
                dtype = dtype
            )
        )
        self._init_weight()
        
    def _init_weight(self):
        nn.init.trunc_normal_(
            self.embedding, 
            mean = 0, 
            std = 1, 
            a = -3, 
            b = 3
        )
    
    def forward(
        self, 
        token_ids: torch.tensor
    ):
        return self.embedding[token_ids]
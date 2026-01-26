import torch
from cs336_basics.module.softmax import softmax

def scaled_dot_product_attention(
    query: torch.tensor,
    key: torch.tensor,
    value: torch.tensor,
    attn_mask:torch.tensor
) -> torch.tensor:
    d_k = query.shape[-1]
    assert d_k == key.shape[-1], "d_k is not equal"
    score = torch.einsum("b ... i d, b ... j d -> b ... i j", query, key) / (d_k ** 0.5)
    if attn_mask is not None:
        score = score.masked_fill(attn_mask == 0, -1e9)
    return torch.einsum("b ... i s, b ... s d -> b ... i d", softmax(score, dim = -1), value)
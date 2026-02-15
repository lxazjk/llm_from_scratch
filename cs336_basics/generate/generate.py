import torch
import torch.nn as nn
from cs336_basics.module.Softmax import softmax

def sampler(
    logits: torch.tensor,
    top_p: float = 1.0,
    temperature: float = 1.0,
):
    logits.div_(temperature)
    probs = softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits.scatter_(dim=-1, index=indices_to_remove, value=float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids

def generate(
    model: nn.Module,
    input_ids: torch.tensor,
    max_seq_len: int,
    special_token: int,
    top_p: float = 1.0,
    temprature: float = 1.0,
):
    batch_size, seq_len = input_ids.shape
    model.eval()
    special_token = torch.tensor([special_token]).to(input_ids.device)
    special_token = special_token.unsqueeze(0).expand(batch_size, -1)
    for _ in range(max_seq_len):
        logits = model(input_ids)
        input_ids = sampler(logits, temperature=temprature, top_p=top_p)
        if input_ids[..., -1] == special_token:
            break
    return input_ids
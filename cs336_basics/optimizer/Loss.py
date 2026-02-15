from torch import logit
import torch
def Cross_Entropy(
    input_ids: torch.tensor, # [batch_size, vocab_size]
    target_ids: torch.tensor # [batch_size]
) -> torch.tensor:
    """
    Cross Entropy Loss:
        don't simply use -log(softmax(input)) which can cause numerical instability
    """
    batch_size, _ = input_ids.shape
    assert batch_size == target_ids.shape[0], "[ERROR] input_ids and target_ids must have the same batch size"
    max_probs = torch.max(input_ids, dim = -1)
    logits = input_ids - max_probs[0].unsqueeze(-1)
    return torch.log(torch.exp(logits).sum(dim = -1)) - logits[torch.arange(batch_size), target_ids]

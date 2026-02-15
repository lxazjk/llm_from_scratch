import torch

def softmax(
    inputs: torch.tensor,
    dim: int
) -> torch.tensor:
    in_type = inputs.dtype
    input_ids = inputs.to(torch.float32)
    max_input = torch.max(input_ids, dim = dim, keepdim = True)[0]
    print(f"max_inputs {max_input}")
    exp_input = torch.exp(input_ids - max_input)
    return (exp_input / torch.sum(exp_input, dim = dim, keepdim = True)).to(in_type)
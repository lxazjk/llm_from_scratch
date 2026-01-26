import torch

def softmax(
    input: torch.tensor,
    dim: int
) -> torch.tensor:
    in_type = input.dtype
    input = input.to(torch.float32)
    # print(f"max_ans {torch.max(input, dim = dim, keepdim = True)}")
    max_input = torch.max(input, dim = dim, keepdim = True)[0]
    exp_input = torch.exp(input - max_input)
    return (exp_input / torch.sum(exp_input, dim = dim, keepdim = True)).to(in_type)
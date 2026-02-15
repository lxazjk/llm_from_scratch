import math
import torch
import torch.nn as nn
from typing import Optional, Callable
import torch.optim as optim

class SGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        assert lr > 0, "[ERROR] lr must be greater than 0"
        super().__init__(params, {"lr": lr})

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("p", 0)
                p.data -= group["lr"] / math.sqrt(1 + t) * p.grad.data
                state["p"] = t + 1
        return loss


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        betas,
        eps,
        weight_decay,
    ):
        super().__init__(params, {"alpha": lr, "betas": betas, "epsilon": eps, "lambda_reg": weight_decay})

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1, beta2 = group["betas"]
            epsilon = group["epsilon"]
            lambda_reg = group["lambda_reg"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                """
                    python get_state normaly will create an copy
                    use in place reference to avoid memory leak
                """
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.grad.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.grad.data)
                    state["time"] = 1
                # exp_avg = state.get("exp_avg", torch.zeros_like(p.grad.data))
                # exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(p.grad.data))
                # t = state.get("t", 1)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                time = state["time"]
                # inplace modify exp_avg and exp_avg_sq
                exp_avg.mul_(beta1).add_(p.grad.data, alpha = (1 - beta1))
                exp_avg_sq.mul_(beta2).add_(p.grad.data ** 2, alpha = (1 - beta2))
                alpha_t = alpha * math.sqrt(1 - math.pow(beta2, time)) / (1 - math.pow(beta1, time))
                p.data -= alpha_t * exp_avg / (torch.sqrt(exp_avg_sq) + epsilon)
                p.data -= alpha * lambda_reg * p.data
                state["time"] += 1
        return loss

if __name__ == "__main__":
    weights = nn.Parameter(5 * torch.rand((10, 10)))
    opt = SGD([weights], lr = 100)
    for i in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
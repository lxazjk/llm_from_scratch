import math
import torch
import timeit
import argparse
import torch.nn as nn
from cs336_systems.kernel.flash_attn_v2.flash_attn_kernel import flash_attn_fwd
from cs336_basics.model import scaled_dot_product_attention

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Scaled Dot Production models')
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence context length")
    parser.add_argument("--d_k", type=int, default=512, help="d_k length")
    parser.add_argument("--d_v", type=int, default=768, help="d_v length")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Number of warmup steps")
    parser.add_argument('--mixed_precision', type=bool, default=False, help='Use mixed percision to run the model')
    return parser.parse_args()

def warm_up(query, key, value, warmup_steps):
    model = SDPA(query, key, value)
    for _ in range(warmup_steps):
        output = model()
        if _ == warmup_steps - 1: forward_output = output
        loss = output.mean()
        loss.backward()
        if _ == warmup_steps - 1: q_grad, k_grad, v_grad = query.grad, key.grad, value.grad
        del query.grad
        del key.grad
        del value.grad
    return forward_output, q_grad, k_grad, v_grad

class SDPA(nn.Module):
    def __init__(self, query, key, value):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value

    def forward(self):
        return scaled_dot_product_attention(self.query, self.key, self.value)

def baseline(query, key, value, num_steps:int = 2):
    forward_time_total = 0
    backward_time_total = 0
    model = SDPA(query, key, value)
    for _ in range(num_steps):
        forward_time = timeit.default_timer()
        output = model()
        torch.cuda.synchronize()
        forward_time_total += timeit.default_timer() - forward_time
        backward_time = timeit.default_timer()
        loss = output.mean().backward()
        torch.cuda.synchronize()
        backward_time_total += timeit.default_timer() - backward_time
        del query.grad
        del key.grad
        del value.grad
    print(f"baseline\nforward_avg {forward_time_total / num_steps}\nbackward_avg {backward_time_total / num_steps}")

    model = torch.compile(model)
    forward_time_total = 0
    backward_time_total = 0
    for _ in range(num_steps):
        forward_time = timeit.default_timer()
        output = model()
        torch.cuda.synchronize()
        forward_time_total += timeit.default_timer() - forward_time
        backward_time = timeit.default_timer()
        loss = output.mean().backward()
        torch.cuda.synchronize()
        backward_time_total += timeit.default_timer() - backward_time
        del query.grad
        del key.grad
        del value.grad
    print(f"torch compile\nforward_avg {forward_time_total / num_steps}\nbackward_avg {backward_time_total / num_steps}")

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    num_heads = args.num_heads
    seq_len = args.seq_len
    d_k = args.d_k
    d_v = args.d_v
    warmup_steps = args.warmup_steps
    # query = torch.randn(batch_size, num_heads, seq_len, d_k, device="cuda", requires_grad=True)
    # key = torch.randn(batch_size, num_heads, seq_len, d_k, device="cuda", requires_grad=True)
    # value = torch.randn(batch_size, num_heads, seq_len, d_v,device="cuda", requires_grad=True)
    # 1. 生成并缩放
    query = torch.randn(batch_size, num_heads, seq_len, d_k, device="cuda") * 10
    key   = torch.randn(batch_size, num_heads, seq_len, d_k, device="cuda") * 10
    value = torch.randn(batch_size, num_heads, seq_len, d_v, device="cuda") * 10

    # 2. 此时再开启梯度
    query.requires_grad_()
    key.requires_grad_()
    value.requires_grad_()

    # 现在 query.grad_fn 应该是 None，它是干净的叶子节点
    L = torch.zeros(batch_size, num_heads, seq_len, device="cuda", requires_grad=True)
    forward_output, q_grad, k_grad, v_grad = warm_up(query, key, value, warmup_steps)
    baseline(query, key, value)

    output = torch.zeros(batch_size * num_heads, seq_len, d_v, device="cuda", requires_grad=True)
    L = torch.empty(batch_size * num_heads, seq_len, dtype=query.dtype, device=query.device)
    flash_attn_fwd(query, key, value, output, L, 1.0 / math.sqrt(d_k), 1, 16, 16)
    output = output.view(batch_size, num_heads, seq_len, d_v)
    # print(output, forward_output)
    print((output - forward_output).abs().max())
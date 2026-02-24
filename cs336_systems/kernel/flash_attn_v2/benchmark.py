import math
import torch
import timeit
import argparse
import torch.nn as nn
from cs336_systems.kernel.flash_attn_v2.flash_attn_fwd import flash_attn_fwd
from cs336_systems.kernel.flash_attn_v2.flash_attn_bwd import flash_attn_bwd
from cs336_basics.model import scaled_dot_product_attention

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Scaled Dot Production models')
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence context length")
    parser.add_argument("--d_k", type=int, default=512, help="d_k length")
    parser.add_argument("--d_v", type=int, default=768, help="d_v length")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
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

def baseline(query, key, value, num_steps:int = 10):
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

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        ctx.save_for_backward(q, k, v)
        ctx.q_shape, ctx.k_shape, ctx.v_shape = q.shape, k.shape, v.shape
        ctx.scale = scale
        q_flat, k_flat, v_flat = q.view(-1, q.shape[1], q.shape[2]), k.view(-1, k.shape[1], k.shape[2]), v.view(-1, v.shape[1], v.shape[2])
        B_H, S, Dk = q_flat.shape
        Dv = v_flat.shape[-1]
        o = torch.empty((B_H, S, Dv), dtype=q.dtype, device=q.device)
        lse = torch.empty((B_H, S), dtype=torch.float32, device=q.device)
        forward_time = timeit.default_timer()
        flash_attn_fwd(q_flat, k_flat, v_flat, o, lse, scale, 1, 16, 16)
        ctx.forward_time = timeit.default_timer() - forward_time
        ctx.save_for_backward(q_flat, k_flat, v_flat, o, lse)
        return o.view(*ctx.q_shape[:-1], Dv)

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do_flat = do.reshape(-1, o.shape[1], o.shape[2])
        backward_time = timeit.default_timer()
        dq, dk, dv = flash_attn_bwd(q, k, v, o, lse, do_flat, ctx.scale, 16, 16)
        ctx.backward_time = timeit.default_timer() - backward_time
        print(f"flash_attn_v2\nforward {ctx.forward_time}\nbackward {ctx.backward_time}")
        return dq.view(ctx.q_shape), dk.view(ctx.k_shape), dv.view(ctx.v_shape), None

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    num_heads = args.num_heads
    seq_len = args.seq_len
    d_k = args.d_k
    d_v = args.d_v
    warmup_steps = args.warmup_steps
    query = torch.randn(batch_size, seq_len, d_k, device="cuda", requires_grad=True)
    key = torch.randn(batch_size, seq_len, d_k, device="cuda", requires_grad=True)
    value = torch.randn(batch_size, seq_len, d_v,device="cuda", requires_grad=True)
    L = torch.zeros(batch_size, seq_len, device="cuda", requires_grad=True)
    forward_output, q_grad, k_grad, v_grad = warm_up(query, key, value, warmup_steps)
    baseline(query, key, value)
    q_triton = query.detach().clone().requires_grad_()
    k_triton = key.detach().clone().requires_grad_()
    v_triton = value.detach().clone().requires_grad_()
    output_triton = FlashAttentionFunction.apply(q_triton, k_triton, v_triton, 1.0 / math.sqrt(d_k))
    loss_triton = output_triton.mean()
    loss_triton.backward()
    print(f"output max error: {(output_triton - forward_output).abs().max()}")
    print(f"dQ max error: {(q_triton.grad - q_grad).abs().max()}")
    print(f"dK max error: {(k_triton.grad - k_grad).abs().max()}")
    print(f"dV max error: {(v_triton.grad - v_grad).abs().max()}")
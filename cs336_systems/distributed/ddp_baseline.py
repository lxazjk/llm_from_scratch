import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
from torch.cuda import nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Transofrmer models')
    parser.add_argument("--d_model", type=int, required=True, help="Model dimension")
    parser.add_argument("--d_ff", type=int, required=True, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")
    parser.add_argument("--context_length", type=int, default=1024, help="Sequence context length")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument('--mixed_precision', type=bool, default=False, help='Use mixed percision to run the model')
    return parser.parse_args()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "39500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # GPU
    print(f"rank {rank} world_size {world_size}")
    torch.cuda.set_device(rank + 4)

def ddp_baseline(model, optimizer, loss_fn, args, rank, world_size):
    for step in range(11):
        optimizer.zero_grad()
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        labels = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        t_comm_start = timeit.default_timer()
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        torch.cuda.synchronize()
        t_comm_end = timeit.default_timer()
        if step == 10 and rank == 0:
            print(f"naive dpp: step {step} loss {loss.item()} comm time {t_comm_end - t_comm_start}")
        optimizer.step()

def ddp_flatten_reduce(model, optimizer, loss_fn, args, rank, world_size):
    for step in range(11):
        optimizer.zero_grad()
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        labels = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        t_grad_start = timeit.default_timer()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size
        with torch.no_grad():
            updated_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
            for old_grad, new_grad in zip(grads, updated_grads):
                old_grad.copy_(new_grad)
        torch.cuda.synchronize()
        t_grad_end = timeit.default_timer()
        if step == 10 and rank == 0:
            print(f"naive ddp with flatten reduce :step {step} loss {loss.item()} comm time {t_grad_end - t_grad_start}")
        optimizer.step()

class ddp_overlap_individual_parameters(nn.Module):
    def __init__(self, module: nn.Module, world_size: int):
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.handles = []
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, p):
        def hook(param):
            handle = dist.all_reduce(param.grad, async_op=True)
            self.handles.append(handle)
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        with torch.no_grad():
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad /= self.world_size

def ddp_overlap_hook(model, optimizer, loss_fn, args, rank, world_size):
    torch.cuda.synchronize()
    for step in range(11):
        optimizer.zero_grad()
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        labels = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        t_grad_start = timeit.default_timer()
        model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        t_grad_end = timeit.default_timer()
        if step == 10 and rank == 0:
            print(f"naive ddp with overlap hook :step {step} loss {loss.item()} comm time {t_grad_end - t_grad_start}")
        optimizer.step()

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Dict

class ddp_overlap_bucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 20, world_size: int = 2):
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.handles = []
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
        self.buckets = []
        self.param_to_bucket_id = {}
        current_bucket = []
        current_bucket_size = 0
        params = list(self.module.parameters())
        for p in reversed(params):
            if not p.requires_grad:
                continue
            p_size = p.numel() * p.element_size()
            current_bucket.append(p)
            current_bucket_size += p_size
            if current_bucket_size >= self.bucket_size_bytes:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0
        if current_bucket:
            self.buckets.append(current_bucket)

        self.bucket_id_to_pending_count = {}
        for b_id, bucket_params in enumerate(self.buckets):
            for p in bucket_params:
                self.param_to_bucket_id[id(p)] = b_id
        self.full_counts = [len(b) for b in self.buckets]
        self.reset_buckets()
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def reset_buckets(self):
        self.bucket_id_to_pending_count = {i: count for i, count in enumerate(self.full_counts)}

    def _make_hook(self, p):
        def hook(param):
            b_id = self.param_to_bucket_id[id(p)]
            self.bucket_id_to_pending_count[b_id] -= 1
            if self.bucket_id_to_pending_count[b_id] == 0:
                self._all_reduce_bucket(b_id)
        return hook

    def _all_reduce_bucket(self, b_id):
        bucket_params = self.buckets[b_id]
        grads = [p.grad for p in bucket_params if p.grad is not None]
        if not grads:
            return
        flat_grad = torch._utils._flatten_dense_tensors(grads)
        handle = dist.all_reduce(flat_grad, async_op=True)
        self.handles.append((handle, flat_grad, grads))

    def forward(self, *inputs, **kwargs):
        self.reset_buckets()
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, flat_grad, grads in self.handles:
            handle.wait()
            flat_grad /= self.world_size
            updated_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)
            for old_grad, new_grad in zip(grads, updated_grads):
                old_grad.copy_(new_grad)
        self.handles.clear()

def ddp_benchmark(rank, world_size, args):
    setup(rank, world_size)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        rope_theta=args.rope_theta,
    ).to(device="cuda")
    # model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    loss_fn = cross_entropy
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    model.train()
    ddp_baseline(model, optimizer, loss_fn, args, rank, world_size)
    ddp_flatten_reduce(model, optimizer, loss_fn, args, rank, world_size)
    Model = ddp_overlap_individual_parameters(model, world_size)
    ddp_overlap_hook(Model, optimizer, loss_fn, args, rank, world_size)
    Model = ddp_overlap_bucketed(model, bucket_size_mb=1.0, world_size=world_size)
    ddp_overlap_hook(Model, optimizer, loss_fn, args, rank, world_size)

if __name__ == "__main__":
    args = parse_args()
    world_size = 4
    mp.spawn(ddp_benchmark, args=(world_size, args), nprocs=world_size, join=True)
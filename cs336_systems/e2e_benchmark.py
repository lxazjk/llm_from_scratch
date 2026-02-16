import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import timeit
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

def warm_up(
    model : nn.Module,
    optimizer : optim.Optimizer,
    loss_fn : nn.Module,
    warmup_steps : int,
):
    model.train()
    for step in range(warmup_steps):
        optimizer.zero_grad()
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        labels = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

def profile_one_epoch_sync(
    model : nn.Module,
    optimizer : optim.Optimizer,
    loss_fn : nn.Module,
    num_steps : int = 10
):
    model.train()
    forward_time_total = 0.0
    loss_time_total = 0.0
    backward_time_total = 0.0
    optimizer_time_total = 0.0
    total_time = timeit.default_timer()
    for _ in range(num_steps):
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        labels = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        torch.cuda.memory._record_memory_history(enabled='all', stacks='all', max_entries=100000)
        forward_time = timeit.default_timer()
        outputs = model(input_ids)
        if _ == num_steps - 1: torch.cuda.memory._dump_snapshot("forward_snapshot.json")
        torch.cuda.memory._record_memory_history(enabled=False)
        torch.cuda.synchronize()
        forward_time_total += timeit.default_timer() - forward_time
        loss_time = timeit.default_timer()
        torch.cuda.memory._record_memory_history(enabled='all', stacks='all', max_entries=100000)
        loss = loss_fn(outputs, labels)
        if _ == num_steps - 1: torch.cuda.memory._dump_snapshot("loss_snapshot.json")
        torch.cuda.memory._record_memory_history(enabled=False)
        torch.cuda.synchronize()
        loss_time_total += timeit.default_timer() - loss_time
        backward_time = timeit.default_timer()
        torch.cuda.memory._record_memory_history(enabled='all', stacks='all', max_entries=100000)
        loss.backward()
        if _ == num_steps - 1: torch.cuda.memory._dump_snapshot("backward_snapshot.json")
        torch.cuda.memory._record_memory_history(enabled=False)
        torch.cuda.synchronize()
        backward_time_total += timeit.default_timer() - backward_time
        optimizer_time = timeit.default_timer()
        torch.cuda.memory._record_memory_history(enabled='all',stacks='all',max_entries=100000)
        optimizer.step()
        if _ == num_steps - 1: torch.cuda.memory._dump_snapshot("optimizer_snapshot.json")
        torch.cuda.memory._record_memory_history(enabled=False)
        torch.cuda.synchronize()
        optimizer_time_total += timeit.default_timer() - optimizer_time
    torch.cuda.synchronize()
    total_time = timeit.default_timer() - total_time
    print(f"Forward time: {forward_time_total / num_steps}")
    print(f"Loss time: {loss_time_total / num_steps}")
    print(f"Backward time: {backward_time_total / num_steps}")
    print(f"Optimizer time: {optimizer_time_total / num_steps}")
    print(f"Total time: {total_time / num_steps}")

def profile_one_epoch_async(
    model : nn.Module,
    optimizer : optim.Optimizer,
    loss_fn : nn.Module,
    num_steps : int = 10
):
    model.train()
    forward_time_total = 0.0
    loss_time_total = 0.0
    backward_time_total = 0.0
    optimizer_time_total = 0.0
    total_time = timeit.default_timer()
    for _ in range(num_steps):
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        labels = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        forward_time = timeit.default_timer()
        outputs = model(input_ids)
        forward_time_total += timeit.default_timer() - forward_time
        loss_time = timeit.default_timer()
        loss = loss_fn(outputs, labels)
        loss_time_total += timeit.default_timer() - loss_time
        backward_time = timeit.default_timer()
        loss.backward()
        backward_time_total += timeit.default_timer() - backward_time
        optimizer_time = timeit.default_timer()
        optimizer.step()
        optimizer_time_total += timeit.default_timer() - optimizer_time
    torch.cuda.synchronize()
    total_time = timeit.default_timer() - total_time
    print(f"Forward time: {forward_time_total / num_steps}")
    print(f"Loss time: {loss_time_total / num_steps}")
    print(f"Backward time: {backward_time_total / num_steps}")
    print(f"Optimizer time: {optimizer_time_total / num_steps}")
    print(f"Total time: {total_time / num_steps}")

if __name__ == "__main__":
    args = parse_args()
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        rope_theta=args.rope_theta,
    ).to(device="cuda")
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    loss_fn = cross_entropy
    warm_up(model, optimizer, loss_fn, args.warmup_steps)
    profile_one_epoch_sync(model, optimizer, loss_fn, num_steps = 10)
    profile_one_epoch_async(model, optimizer, loss_fn, num_steps = 10)
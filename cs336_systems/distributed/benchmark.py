import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import matplotlib.pyplot as plt

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505" # 换个端口防止冲突
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        # 注意：这里改成了 % 逻辑，防止 rank+1 溢出
        torch.cuda.set_device(rank + 1)

def benchmark_run(rank, world_size, backend, size_mb, return_dict):
    setup(rank, world_size, backend)
    num_elements = (size_mb * 1024 * 1024) // 4
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if backend == "nccl" else "cpu")

    data = torch.randn(num_elements, dtype=torch.float32, device=device)
    for _ in range(3):
        dist.all_reduce(data)
    iters = 10
    if backend == "nccl":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            dist.all_reduce(data)
        end_event.record()
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / (iters)
    else:
        start_time = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(data)
        avg_time = (time.perf_counter() - start_time) / iters

    if rank == 0:
        return_dict[(backend, world_size, size_mb)] = avg_time
        print(f"Finished: {backend}, WS={world_size}, Size={size_mb}MB, Time={avg_time:.4f}ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    backends = ["gloo", "nccl"]
    world_sizes = [2, 4, 6]
    sizes_mb = [1, 10, 100, 1000]

    manager = mp.Manager()
    results_dict = manager.dict()

    for backend in backends:
        for ws in world_sizes:
            if backend == "nccl" and ws > torch.cuda.device_count():
                print(f"Skipping NCCL WS={ws} (insufficient GPUs)")
                continue
            for size in sizes_mb:
                mp.spawn(benchmark_run, args=(ws, backend, size, results_dict), nprocs=ws, join=True)

    print("\nGenerating Plot...")
    plt.figure(figsize=(10, 6))
    for backend in backends:
        for ws in world_sizes:
            x_sizes = []
            y_times = []
            for size in sizes_mb:
                key = (backend, ws, size)
                if key in results_dict:
                    x_sizes.append(size)
                    y_times.append(results_dict[key])
            if x_sizes:
                plt.plot(x_sizes, y_times, marker='o', label=f"{backend.upper()}, WS={ws}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Data Size (MB)')
    plt.ylabel('Avg Time (seconds)')
    plt.title('All-Reduce Benchmark: Gloo vs NCCL')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    plt.savefig('benchmark_plot.png')
    print("Plot saved as benchmark_plot.png")
    plt.show()
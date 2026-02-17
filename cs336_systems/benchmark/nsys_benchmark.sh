uv run nsys profile -o result --force-overwrite true --python-backtrace=cuda --pytorch=autograd-shapes-nvtx python cs336_systems/benchmark/e2e_benchmark.py --d_model 768 --context_length 128 --d_ff 3072 --num_heads 12 --num_layers 12



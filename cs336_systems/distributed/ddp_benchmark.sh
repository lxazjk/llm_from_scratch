echo "Small model"
uv run cs336_systems/distributed/ddp_baseline.py  --d_model 768 --context_length 128 --d_ff 3072 --num_heads 12 --num_layers 12 # small

# echo "Medium model"
# uv run cs336_systems/benchmark/e2e_benchmark.py --d_model 1024 --context_length 128 --d_ff 4096 --num_heads 16 --num_layers 24 # medium

# echo "large model"
# uv run cs336_systems/benchmark/e2e_benchmark.py --d_model 1280 --context_length 128 --d_ff 5120 --num_heads 20 --num_layers 32 # large

# echo "XL model"
# uv run cs336_systems/benchmark/e2e_benchmark.py --d_model 1600 --context_length 128 --d_ff 6400 --num_heads 25 --num_layers 48 # extra large

# echo "2.7B"
# uv run cs336_systems/benchmark/e2e_benchmark.py --d_model 2560 --context_length 128 --d_ff 10240 --num_heads 32 --num_layers 32 # 2.7B

import torch
from cs336_basics.module.Transformer_LM import Transformer_LM
from cs336_basics.optimizer.SGD import AdamW
from cs336_basics.training.train import train
from cs336_basics.training.dataloader import data_loader
from cs336_basics.optimizer.Loss import Cross_Entropy

if __name__ == "__main__":
    vocab_size = 10000
    context_length = 768
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    model = Transformer_LM(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads,
        d_ff = d_ff,
        rope_theta = rope_theta,
        device = device,
        dtype = dtype
    )
    optimizer = AdamW(
        model.parameters(),
        lr = 1e-3,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.01
    )
    train(
        model = model,
        optim = optimizer,
        dataset_dir = "data/TinyStoriesV2-GPT4-train.bin",
        num_step = 1000,
        batch_size = 128,
        context_length = context_length,
        device = device,
        Loss = Cross_Entropy,
        data_loader = data_loader
    )
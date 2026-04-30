import os
import torch
import torch.distributed as dist

def setup(rank, world_size, device="cpu"):
    if device != "cpu":
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    backend = "gloo" if device == "cpu" else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

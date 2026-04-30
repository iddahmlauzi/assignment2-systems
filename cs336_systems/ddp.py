# Distributed Data Parallel Training
import torch
from typing import type
from copy import deepcopy
from cs336_systems.utils import setup
import torch.multiprocessing as mp
import torch.distributed as dist


def distributed_data_parallel_training(model_class, world_size=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp.spawn(
        _distributed_data_parallel_training,
        args=(world_size, device, model_class),
        nprocs=world_size,
        join=True,
    )


def _distributed_data_parallel_training(
    rank: int, 
    world_size: int,
    device: str,
    model_class: type[torch.nn.Module]
):
    setup(rank, world_size, device=device)
    
    # Make sure each processes finishes running its setup
    dist.barrier()
    torch.manual_seed(rank) # So each rank is iniralized with a different model
    
    base_model = model_class().to(device)
    distributed_model = deepcopy(base_model)
    

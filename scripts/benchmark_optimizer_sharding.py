import modal
import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import statistics
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
from itertools import product
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_systems.utils import setup
from cs336_systems.ddp import NaiveDDP, DDP
from cs336_systems.optim import ShardedOptimizer
from cs336_basics.model import TransformerLM
from cs336_basics.loss import cross_entropy
from cs336_basics.optim import AdamW



wandb_secret = modal.Secret.from_name("wandb")


def run_training_step(model, optimizer, x, y, mixed_precision: bool=False):
    """Runs a single training step based on the provided mode"""
    
    autocast = torch.autocast(device_type=x.device.type, dtype=torch.bfloat16) if mixed_precision else nullcontext()
    # Forward pass
    with autocast:
        logits = model(x)
        
    # Backward Pass
    loss = cross_entropy(inputs=logits, targets=y)
    loss.backward()
    model.finish_gradient_synchronization()
        
    # Optimizer Step
    optimizer.step()
    

def benchmark(rank, world_size, 
                    device="cpu", 
                    x=None, y=None,
                    vocab_size: int=10000,
                    context_length: int=512,
                    batch_size: int=4,
                    optim_shard: bool = False,
                    d_model: int=2560,
                    num_layers: int=32,
                    num_heads: int=32,
                    d_ff: int=10240,
                    warmup_iters=5, num_trials=10
                    ):
    
    setup(rank, world_size, device)
    
    # 1. Initialize transformer model and optimizer
    model = TransformerLM(vocab_size=vocab_size,
                          context_length=context_length,
                          d_model=d_model,
                          num_layers=num_layers,
                          num_heads=num_heads,
                          d_ff=d_ff,
                          device=device)
    
    model = DDP(model)
    if optim_shard:
        optimizer = ShardedOptimizer(params=model.parameters(), 
                                     optimizer_cls=AdamW)
    else:
        optimizer = AdamW(params=model.parameters())
        
    torch.cuda.synchronize()
    mem_after_init = torch.cuda.memory_allocated()

    
    # 2. Get current batch of data
    local_bs = batch_size // world_size
    x = x[rank * local_bs : (rank + 1) * local_bs].to(device)
    y = y[rank * local_bs : (rank + 1) * local_bs].to(device)
    
    
    # Run warmup
    for _ in range(warmup_iters):
        run_training_step(model, optimizer, x, y)
        optimizer.zero_grad(set_to_none=True)
    
    # Benchmark Full training Step
    times: list[float] = [] 
    for _ in range(num_trials):
        start_end = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_end.record()
        run_training_step(model, optimizer, x, y)
        end_event.record()
        
        torch.cuda.synchronize() 
        times.append(start_end.elapsed_time(end_event))
        optimizer.zero_grad(set_to_none=True)
        
    # Benchmark Memory Use --> Should I average these? I don't feel like it.
    logits = model(x)
    loss = cross_entropy(inputs=logits, targets=y)
    loss.backward()
    model.finish_gradient_synchronization()

    torch.cuda.synchronize()
    mem_before_optim = torch.cuda.memory_allocated()

    optimizer.step()

    torch.cuda.synchronize()
    mem_after_optim = torch.cuda.memory_allocated()
                
    
    # Gather results from all ranks onto every rank
    all_times = [None] * world_size
    dist.all_gather_object(all_times, times)
    
    all_mem_after_init = [None] * world_size
    all_mem_before_optim = [None] * world_size  
    all_mem_after_optim = [None] * world_size

    dist.all_gather_object(all_mem_after_init, mem_after_init)
    dist.all_gather_object(all_mem_before_optim, mem_before_optim)
    dist.all_gather_object(all_mem_after_optim, mem_after_optim)
    
    if rank == 0:
        times = [t for step_times in all_times for t in step_times]
        optimizer_type = "sharded" if optim_shard else "base"
        final = {
            "Optimizer Type": optimizer_type,
            "Time Per Training Step": f"{statistics.mean(times):.3f} ± {statistics.stdev(times):3f}",
            "Mem After Init Rank 0 (GB)": all_mem_after_init[0] / (1024**3),
            "Mem After Init Rank 1 (GB)": all_mem_after_init[1] / (1024**3),
            "Mem Before Optim Rank 0 (GB)": all_mem_before_optim[0] / (1024**3),
            "Mem Before Optim Rank 1 (GB)": all_mem_before_optim[1] / (1024**3),
            "Mem After Optim Rank 0(GB)": all_mem_after_optim[0] / (1024**3),
            "Mem After Optim Rank 1 (GB)": all_mem_after_optim[1] / (1024**3),
        }
        with open("/tmp/result.json", "w") as f:
            json.dump(final, f)
    
    del x, y, model, optimizer
    torch.cuda.empty_cache()
    dist.destroy_process_group()

        

@app.function(image=build_image(), 
              volumes=VOLUME_MOUNTS, 
              max_containers=1,
              gpu="B200:2",
              secrets=[wandb_secret],
              timeout=2700)
def run_benchmark(world_size: int=2, vocab_size = 10000, batch_size = 4, context_length = 512, optim_shard=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
    
    mp.spawn(fn=benchmark, args=(world_size, device, x, y, vocab_size, context_length, batch_size, optim_shard), nprocs=world_size, join=True)
    with open("/tmp/result.json") as f:
        return json.load(f)
    

@app.local_entrypoint()
def main():
    
    results = []
    for optim_shard in [False, True]:
        print(f"Benchmarking Optimizer State Sharding with optim_shard={optim_shard}....")
        result = run_benchmark.remote(optim_shard=optim_shard)
        results.append(result)
        
    # Turn to a pandas dataframe 
    df = pd.DataFrame(results)
    print(df.to_latex(index=False, column_format="l" + "r" * (len(df.columns) - 1)))
        
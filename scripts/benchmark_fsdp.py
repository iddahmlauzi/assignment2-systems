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
from cs336_systems.fsdp import FSDP
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
                        d_model: int=2560,
                        num_layers: int=32,
                        num_heads: int=32,
                        d_ff: int=10240,
                        warmup_iters=5, num_trials=10,):
    
    setup(rank, world_size, device)
    
    # 1. Initialize transformer model and optimizer
    model = TransformerLM(vocab_size=vocab_size,
                          context_length=context_length,
                          d_model=d_model,
                          num_layers=num_layers,
                          num_heads=num_heads,
                          d_ff=d_ff,
                          device=device)
    
    model = FSDP(model)
    optimizer = ShardedOptimizer(params=model.parameters(), 
                                     optimizer_cls=AdamW)
    
    # 2. Get current batch of data
    local_bs = batch_size // world_size
    x = x[rank * local_bs : (rank + 1) * local_bs].to(device)
    y = y[rank * local_bs : (rank + 1) * local_bs].to(device)
    
    
    # Run warmup
    for _ in range(warmup_iters):
        run_training_step(model, optimizer, x, y)
        optimizer.zero_grad(set_to_none=True)
    
    # Benchmark Full training Step
    train_step_times: list[float] = [] 
    grad_sync_times: list[float] = [] 
    for _ in range(num_trials):
        step_start_event = torch.cuda.Event(enable_timing=True)
        step_end_event = torch.cuda.Event(enable_timing=True)
        
        sync_start_event = torch.cuda.Event(enable_timing=True)
        sync_end_event = torch.cuda.Event(enable_timing=True)
        
        step_start_event.record()
        with nvtx.range("forward"):
            logits = model(x)
        
        with nvtx.range("backward"):
            loss = cross_entropy(inputs=logits, targets=y)
            loss.backward()
        
        sync_start_event.record()
        with nvtx.range("grad_sync"):
            model.finish_gradient_synchronization()
        sync_end_event.record()
            
        with nvtx.range("optimizer"):
            optimizer.step()
        
        step_end_event.record()
        
        torch.cuda.synchronize() 
        train_step_times.append(step_start_event.elapsed_time(step_end_event))
        grad_sync_times.append(sync_start_event.elapsed_time(sync_end_event))
        optimizer.zero_grad(set_to_none=True)
        
    
    # Gather results from all ranks onto every rank
    all_train_step_times = [None] * world_size
    all_grad_sync_times = [None] * world_size
    dist.all_gather_object(all_train_step_times, train_step_times)
    dist.all_gather_object(all_grad_sync_times, grad_sync_times)
    
    if rank == 0:
        train_step_times = [t for step_times in all_train_step_times for t in step_times]
        grad_times = [t for sync_times in all_grad_sync_times for t in sync_times]
        final = {
            "Time Per Training Step": f"{statistics.mean(train_step_times):.3f} ± {statistics.stdev(train_step_times):3f}",
            "Time to communicate gradients": f"{statistics.mean(grad_times):.3f} ± {statistics.stdev(grad_times):3f}",
            "Proportion on gradient Communication": f"{statistics.mean(grad_times) / statistics.mean(train_step_times)}"
            
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
def run_ddp(world_size: int=2, vocab_size = 10000, batch_size = 4, context_length = 512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
    
    mp.spawn(fn=benchmark, args=(world_size, device, x, y, vocab_size, context_length, batch_size), nprocs=world_size, join=True)
    with open("/tmp/result.json") as f:
        return json.load(f)
    
    
@app.function(image=build_image(), 
              volumes=VOLUME_MOUNTS, 
              max_containers=1,
              gpu="B200:2",
              secrets=[wandb_secret],
              timeout=2700)
def run_ddp_nsys():
    import subprocess
    os.makedirs("/root/data/ddp_profile", exist_ok=True)
    cmd = [
        "nsys", "profile",
        "-o", "/root/data/fsdp_profile",
        "--force-overwrite", "true",
        "--pytorch", "autograd-nvtx",
        "--trace=cuda,nvtx",
        "python", __file__
    ]
    print(f"Launching Profiler: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

@app.local_entrypoint()
def main(profile: bool = False):
    
    print(f"Benchmarking FSDP....")
    if profile:
        run_ddp_nsys.remote()
    else:
        result = run_ddp.remote()
        print(result)
        
    
if __name__ == "__main__":
    world_size = 2
    vocab_size = 10000
    batch_size = 4
    context_length = 512
    
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))
    
    mp.spawn(fn=benchmark, args=(world_size, "cuda", x, y, vocab_size, context_length, batch_size), nprocs=world_size, join=True)
import modal
import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import statistics
from itertools import product
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_systems.utils import setup


wandb_secret = modal.Secret.from_name("wandb")

def benchmark_all_reduce(rank, world_size, data_mb=1, device="cpu", warmup_iters=5, num_trials=10):
    setup(rank, world_size, device)
    n = int(data_mb * 1e6 / 4) # 4 bytes per torch float number
    data = torch.rand(n).to(device)
    
    # Run warmup
    for _ in range(warmup_iters):
        dist.all_reduce(data, async_op=False)
    
    # Time Benchmarking
    # Note this keeps accumulating the same data tensor. Could make a new one technically but this might be unnecessary
    times: list[float] = [] 
    for _ in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        dist.all_reduce(data, async_op=False)
        end_event.record()
        
        torch.cuda.synchronize() 
        times.append(start_event.elapsed_time(end_event))
        
    
    # Gather results from all ranks onto every rank
    all_results = [None] * world_size
    dist.all_gather_object(all_results, times)
    
    
    if rank == 0:
        all_times = [t for rank_times in all_results for t in rank_times]
        final = {
            "Num Processes": world_size,
            "Data Size (MB)": data_mb,
            "Time": f"{statistics.mean(all_times):.3f} ± {statistics.stdev(all_times):3f}"
        }
        with open("/tmp/result.json", "w") as f:
            json.dump(final, f)
    
    del data
    torch.cuda.empty_cache()
    dist.destroy_process_group()

        

@app.function(image=build_image(), 
              volumes=VOLUME_MOUNTS, 
              max_containers=1,
              gpu="B200:6",
              secrets=[wandb_secret],
              timeout=2700)
def run_distributed(world_size: int=4, data_mb=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp.spawn(fn=benchmark_all_reduce, args=(world_size, data_mb, device), nprocs=world_size, join=True)
    with open("/tmp/result.json") as f:
        return json.load(f)

@app.local_entrypoint()
def main():
    num_processes = [2, 4, 6]
    data_sizes = [1, 10, 100, 1000]
    
    results = []
    for world_size, data_mb in product(num_processes, data_sizes):
        print(f"Benchmarking all reduce with world_size={world_size}, data={data_mb}....")
        result = run_distributed.remote(world_size=world_size, data_mb=data_mb)
        results.append(result)
        
    # Turn to a pandas dataframe 
    df = pd.DataFrame(results)
    df["Num Processes"] = df["Num Processes"].astype(int)
    df["Data Size (MB)"] = df["Data Size (MB)"].astype(int)
    print(df.to_latex(index=False, column_format="lrr")) 
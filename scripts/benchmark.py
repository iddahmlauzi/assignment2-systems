import argparse
import copy
import modal
import timeit
import os
import torch
import statistics
import pandas as pd
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
from typing import Literal
from types import SimpleNamespace
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image, user_volume
from cs336_basics.model import TransformerLM
from cs336_basics.loss import cross_entropy
from cs336_basics.optim import AdamW

wandb_secret = modal.Secret.from_name("wandb")


# Table 1: Specifications of different model sizes
GPT_2_CONFIGS = {'small': {'d_model': 768,  'd_ff': 3072,  'num_layers': 12, 'num_heads': 12}, 
                'medium': {'d_model': 1024, 'd_ff': 4096,  'num_layers': 24, 'num_heads': 16}, 
                'large':  {'d_model': 1280, 'd_ff': 5120,  'num_layers': 36, 'num_heads': 20}, 
                'xl':     {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32}, 
                '10B':    {'d_model': 4608, 'd_ff': 12288, 'num_layers': 50, 'num_heads': 36}}

MODE_TO_STEP_NAME = {
    "f": "forward",
    "fb": "backward",
    "fbo": "optimizer"
}
    

def run_training_step(model, optimizer, x, y, mode, mixed_precision: bool=False, checkpoint_k=1):
    """Runs a single training step based on the provided mode"""
    
    # In forward only mode, we want to remove gradient tracking and other 
    # Overheads used by autograd
    ctx = torch.inference_mode() if mode == "f" else nullcontext()
    autocast = torch.autocast(device_type=x.device.type, dtype=torch.bfloat16) if mixed_precision else nullcontext()
    with ctx:
        # Forward pass
        with nvtx.range("Forward Pass"):
            with autocast:
                logits = model(x, checkpoint_k)
            torch.cuda.synchronize() 
            
        # Backward Pass
        if "b" in mode:
            with nvtx.range("Backward Pass"):
                loss = cross_entropy(inputs=logits, targets=y)
                loss.backward()
                torch.cuda.synchronize() 
        
    # Optimizer Step
    if "o" in mode:
        with nvtx.range("Optimizer Step"):
            optimizer.step()
            torch.cuda.synchronize()
    

@app.function(image=build_image(), 
              volumes=VOLUME_MOUNTS, 
              max_containers=1,
              gpu="B200", 
              secrets=[wandb_secret],
              timeout=2700)
def benchmark(args):
    """Benchmark the model training steps"""
    
    # Any files saved for this will use this name
    save_name = f"{MODE_TO_STEP_NAME[args.mode]}_dmodel_{args.d_model}_dff_{args.d_ff}_layers_{args.num_layers}_heads_{args.num_heads}_context_{args.context_length}"
    
    # nsys profiling
    if hasattr(args, "nsys_profile") and args.nsys_profile:
        import subprocess
        
        
        profile_report_path = f"/root/data/profiling/{save_name}"
        os.makedirs("/root/data/profiling", exist_ok=True)
        
        
        cmd = [
            "nsys", "profile", 
            "--cuda-memory-usage=true",
            "-o", profile_report_path,
            "--force-overwrite", "true",
            "--pytorch", "autograd-nvtx",
            "--trace=cuda,nvtx",
            "env",
            "PYTORCH_ALLOC_CONF=backend:cudaMallocAsync",
            "PYTORCH_NO_CUDA_MEMORY_CACHING=1",
            "python", __file__, 
            "--mode", args.mode, 
            "--vocab_size", str(args.vocab_size),
            "--context_length", str(args.context_length),
            "--d_model", str(args.d_model),
            "--num_layers", str(args.num_layers),
            "--num_heads", str(args.num_heads),
            "--d_ff", str(args.d_ff),
            "--batch_size", str(args.batch_size),
            "--warmup_steps", str(args.warmup_steps),
            "--num_trials", str(args.num_trials),
            "--checkpoint_k", str(args.checkpoint_k),
            "--seed", str(args.seed)
        ]
        if args.mixed_precision:
            cmd.append("--mixed_precision")
        
        print(f"Launching Profiler: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return 0.0, 0.0 # return dummy data
    
    
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize transformer model and optimizer
    model = TransformerLM(vocab_size=args.vocab_size,
                          context_length=args.context_length,
                          d_model=args.d_model,
                          num_layers=args.num_layers,
                          num_heads=args.num_heads,
                          d_ff=args.d_ff,
                          device=device)
    
    if args.compile:
        model = torch.compile(model)
    
    optimizer = AdamW(params=model.parameters(), device=device)
    
    # 2. Create a random batch of data
    x = torch.randint(low=0, high=args.vocab_size, size=(args.batch_size, args.context_length), device=device)
    y = torch.randint(low=0, high=args.vocab_size, size=(args.batch_size, args.context_length), device=device)
    
    
    # 3. Warmup
    for _ in range(args.warmup_steps):
        run_training_step(model, optimizer, x, y, args.mode, args.mixed_precision, args.checkpoint_k)
        
    torch.cuda.synchronize()
    
    # 4. Memory profiling
    # We do checkpoint k if we are memory profiling
    if hasattr(args, "memory_profile") and args.memory_profile:
        # Start recording memory history.
        memory_profile_path = f"/root/data/memory_profiling/{save_name}.pkl"
        os.makedirs("/root/data/memory_profiling", exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        torch.cuda.reset_peak_memory_stats()
        
        run_training_step(model, optimizer, x, y, args.mode, args.mixed_precision, args.checkpoint_k)
        optimizer.zero_grad(set_to_none=True)
        # Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._dump_snapshot(memory_profile_path)
        # Stop recording history.
        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.memory._record_memory_history(enabled=None)
        
        return max_memory / (1024**3), 0.0 # return dummy data
    
    # 5. Time Benchmarking
    times: list[float] = [] 
    with torch.cuda.nvtx.range("measurement_stage"):
        for _ in range(args.num_trials):        
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            run_training_step(model, optimizer, x, y, args.mode, args.mixed_precision, args.checkpoint_k)
            end_event.record()
            
            torch.cuda.synchronize() 
            times.append(start_event.elapsed_time(end_event))
            
            # This might not be necessary since we are just timing
            optimizer.zero_grad(set_to_none=True)
        
    mean_time = statistics.mean(times)
    std = statistics.stdev(times)
    
    del x, y, model, optimizer
    torch.cuda.empty_cache()
    
    return mean_time, std
        
    

@app.local_entrypoint()
def main(
    mode: Literal["f", "fb", "fbo"] = "f", # only forward, forward & backward, forward & backward w/ optimizer step
    vocab_size: int=10000,
    context_length: int=512,
    d_model: int=768,
    num_layers: int=12,
    num_heads: int=12,
    d_ff: int=3072,
    batch_size: int=4,
    warmup_steps: int=5,
    num_trials: int=10, 
    seed: int=42,
    run_gpt_configs: bool=False,
    nsys_profile: bool=False,
    mixed_precision: bool=False,
    memory_profile: bool=False,
    checkpoint_k: int=1,
    compile: bool=False
):
    # Turning the parameters into a Namespace Object
    args = SimpleNamespace(**locals())
    
    if args.nsys_profile:
        print("Starting Remote Profiling Run...")
        benchmark.remote(args)
        
    elif args.run_gpt_configs:
        results = []
        for size in GPT_2_CONFIGS:
            print(f"Benchmarking model of size: {size}")
            size_dict = {}
            size_dict["size"] = size
            
            # Make config for given model size
            model_args = copy.deepcopy(args)
            model_args.d_model = GPT_2_CONFIGS[size]["d_model"]
            model_args.d_ff = GPT_2_CONFIGS[size]["d_ff"]
            model_args.num_layers = GPT_2_CONFIGS[size]["num_layers"]
            model_args.num_heads = GPT_2_CONFIGS[size]["num_heads"]
            
            for mode in ["f", "fb", "fbo"]:
                # Benchmark model
                model_args.mode = mode
                try:
                    average_time, std = benchmark.remote(model_args)
                    size_dict[MODE_TO_STEP_NAME[mode]] = f"{average_time:.2f} ± {std:.2f}"
                    print(f"    - {MODE_TO_STEP_NAME[mode]} : {average_time:.2f} ± {std:.2f} ")
                except Exception as e:
                    print(f"    - {MODE_TO_STEP_NAME[mode]} : failed ({e})")
                    size_dict[MODE_TO_STEP_NAME[mode]] = "OOM"
 
                
            results.append(size_dict)
        
        # Turn to a pandas dataframe 
        df = pd.DataFrame(results)
        print(df.to_latex(index=False, column_format="lrrr"))     
    else:
        print("Benchmarking provided model......")
        average_time, std = benchmark.remote(args)
        if args.memory_profile:
            print(f"Peak memory: {average_time} GB")
        else:
            print(f"Average time: {average_time:.2f} ± {std:.2f}")
        

# Annpoying that I also need this but this is for running with standard python (not modal)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="f")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--checkpoint_k", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    
    args = parser.parse_args()
    benchmark.local(args)

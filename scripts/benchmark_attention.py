# Script for benchmarking pytorch attention
import modal
import torch
import statistics
import pandas as pd
from itertools import product
from cs336_basics.layers import scaled_dot_product_attention
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image

wandb_secret = modal.Secret.from_name("wandb")


@app.cls(
    image=build_image(), 
    volumes=VOLUME_MOUNTS, 
    gpu="B200", 
    secrets=[wandb_secret],
    scaledown_window=3000
)
class Benchmarker:
    @modal.enter()
    def compile_attention(self):
        self.compiled_attn = torch.compile(scaled_dot_product_attention)
        self.base_attn = scaled_dot_product_attention
    
    @modal.method()
    def run_attention(self, batch_size: int=8,
                    d_model: int = 16,
                    context_length: int = 256,
                    warmup_iters: int = 10,
                    num_trials: int = 100, 
                    compile: bool=False):
        """Benchmark the model training steps"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Q = K = V = out = loss = mask = None
        attn_fn = self.compiled_attn if compile else self.base_attn
        
        try:
            Q = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)
            K = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)
            V = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device)
            
            mask = torch.tril(torch.ones(context_length, context_length, device=device, dtype=torch.bool))
            
            for _ in range(warmup_iters):
                attn_fn(Q, K, V, mask=mask)
                
            torch.cuda.synchronize()
                
            # Time 100 forward passes
            forward_times: list[float] = [] 
            for _ in range(num_trials):
                start_event = torch.cuda.Event(enable_timing=True) 
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record() # Begin timing
                out = attn_fn(Q, K, V, mask=mask)
                end_event.record() # Stop timing
                
                torch.cuda.synchronize() 
                forward_times.append(start_event.elapsed_time(end_event))
            
        
            # How much memory is allocated just before the backward pass.
            torch.cuda.reset_peak_memory_stats()
            out = attn_fn(Q, K, V, mask=mask)
            torch.cuda.synchronize()
            memory_before_backward = torch.cuda.memory_allocated()  # currently allocated memory
            
            # Time 100 backward passes
            backward_times: list[float] = [] 
            for _ in range(num_trials):
                start_event = torch.cuda.Event(enable_timing=True) 
                end_event = torch.cuda.Event(enable_timing=True)
                
                out = attn_fn(Q, K, V, mask=mask)
                loss = out.sum()
                torch.cuda.synchronize()
                
                start_event.record() # Begin timing
                loss.backward()
                end_event.record() # stop timing
                
                torch.cuda.synchronize()
                backward_times.append(start_event.elapsed_time(end_event))
                
                # Clear gradients for next iteration
                Q.grad = None
                K.grad = None
                V.grad = None
                
            
            mean_forward_time = statistics.mean(forward_times)
            forward_std = statistics.stdev(forward_times)
            
            mean_backward_time = statistics.mean(backward_times)
            backward_std = statistics.stdev(backward_times)
            
            

            
            result = {
                "d_model": d_model,
                "context_length": context_length,
                "Forward pass time": f"{mean_forward_time:.2f} ± {forward_std:.2f}",
                "Backward pass time": f"{mean_backward_time:.2f} ± {backward_std:.2f}",
                "Memory before backward": f"{memory_before_backward / (1024**3):.4f} GB"
            }
            
            return result
        
        finally:
            del Q, K, V, loss, out, mask
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
                

@app.local_entrypoint()
def main(compile: bool=False
):
    head_dimensions = [16, 32, 64, 128]
    context_lengths = [256, 1024, 4096, 8192, 32768]
    
    benchmarker = Benchmarker()
    
    results = []
    for d_model, context_length in product(head_dimensions, context_lengths):
        print(f"Benchmarking model with d_model={d_model}, context_length={context_length}...")
        try:
            result = benchmarker.run_attention.remote(d_model=d_model, context_length=context_length, compile=compile)
        except Exception as e:
            print(f"    - d_model: {d_model}, context_length: {context_length} : failed ({e})")
            result =  {
                "d_model": d_model,
                "context_length": context_length,
                "Forward pass time": "OOM",
                "Backward pass time": "OOM",
                "Memory before backward": "OOM"
            }
        results.append(result)
        
    # Turn to a pandas dataframe 
    df = pd.DataFrame(results)
    df["d_model"] = df["d_model"].astype(int)
    df["context_length"] = df["context_length"].astype(int)
    print(df.to_latex(index=False, column_format="lrrrr")) 
        
    
        
        
    
    
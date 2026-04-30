import modal
import torch
import statistics
import pandas as pd
from itertools import product
from cs336_basics.layers import scaled_dot_product_attention
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image

wandb_secret = modal.Secret.from_name("wandb")

dtype_map = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16
}


@app.function(
    image=build_image(), 
    volumes=VOLUME_MOUNTS, 
    gpu="B200", 
    secrets=[wandb_secret],
    scaledown_window=3000
)
def benchmark_flash_attention(batch_size: int=1,
                    d_model: int = 16,
                    context_length: int = 256,
                    is_causal: bool = True,
                    dtype: str="float32"):
    """Benchmark the triton flash attention implementation"""
    import triton
    from cs336_systems.flash_attention import FlashAttentionTriton
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    str_dtype = dtype
    dtype = dtype_map[dtype]
    Q = K = V = out = dO = None
    
    try:
        Q = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device, dtype=dtype)
        K = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device, dtype=dtype)
        V = torch.randn(batch_size, context_length, d_model, requires_grad=True, device=device, dtype=dtype)
        dO = torch.randn(batch_size, context_length, d_model, requires_grad=False, device=device, dtype=dtype)
        
        # Benchmark the forward pass
        pytorch_fwd = triton.testing.do_bench(lambda: scaled_dot_product_attention(Q, K, V))
        triton_fwd = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(Q, K, V, is_causal))
        
        # Benchmark the backward pass
        out = FlashAttentionTriton.apply(Q, K, V, is_causal)
        triton_bwd = triton.testing.do_bench(lambda: out.backward(dO, retain_graph=True))
        
        out = scaled_dot_product_attention(Q, K, V)
        pytorch_bwd =  triton.testing.do_bench(lambda: out.backward(dO, retain_graph=True))
        
        # Benchmark end-to-end 
        def triton_fwd_bwd():
            Q.grad = K.grad = V.grad = None
            o = FlashAttentionTriton.apply(Q, K, V, is_causal)
            o.backward(dO)
            
        def pytorch_fwd_bwd():
            Q.grad = K.grad = V.grad = None
            o = scaled_dot_product_attention(Q, K, V)
            o.backward(dO)
               
        triton_e2e = triton.testing.do_bench(triton_fwd_bwd)
        pytorch_e2e = triton.testing.do_bench(pytorch_fwd_bwd)
        
        
        result =  {
                "d_model": d_model,
                "context_length": context_length,
                "dtype": str_dtype,
                "Pytorch Fwd": pytorch_fwd,
                "Triton Fwd": triton_fwd,
                "Pytorch Bwd": pytorch_bwd,
                "Triton Bwd": triton_bwd,
                "Pytorch e2e": pytorch_e2e,
                "Triton e2e": triton_e2e
        }
        return result
               
        
    finally:
            del Q, K, V, out, dO
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
@app.local_entrypoint()
def main():
    head_dimensions = [16, 32, 64, 128]
    context_lengths = [128, 256, 1024, 4096, 8192, 32768, 65536]
    dtypes = ["float32", "bfloat16"]
    
    results = []
    for d_model, context_length, dtype in product(head_dimensions, context_lengths, dtypes):
        print(f"Benchmarking model with d_model={d_model}, context_length={context_length}, dtype={dtype}...")
        try:
            result = benchmark_flash_attention.remote(d_model=d_model, context_length=context_length, dtype=dtype)
        except Exception as e:
            print(f"    - d_model: {d_model}, context_length: {context_length} : failed ({e})")
            result =  {
                    "d_model": d_model,
                    "context_length": context_length,
                    "dtype": dtype,
                    "Pytorch Fwd": "OOM",
                    "Triton Fwd": "OOM",
                    "Pytorch Bwd": "OOM",
                    "Triton Bwd": "OOM",
                    "Pytorch e2e": "OOM",
                    "Triton e2e": "OOM"
            }
        results.append(result)
        
    # Turn to a pandas dataframe 
    df = pd.DataFrame(results)
    df["d_model"] = df["d_model"].astype(int)
    df["context_length"] = df["context_length"].astype(int)
    
    print(
        df.style
        .format(precision=4)
        .hide(axis="index")
        .to_latex(
            column_format="lrrrrrrrr", 
            hrules=True
        )
    )
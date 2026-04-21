import argparse
import modal
from typing import Literal
from types import SimpleNamespace
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image

wandb_secret = modal.Secret.from_name("wandb")

@app.function(image=build_image(), 
              volumes=VOLUME_MOUNTS, 
              max_containers=1,
              gpu="B200", 
              secrets=[wandb_secret],
              timeout=2700)
def benchmark(args):
    print(f"The vocab size is: {args.vocab_size}")


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
    warmup_steps: int=100,
    seed: int=42
):
    # Turning the parameters into a Namespace Object
    args = SimpleNamespace(**locals())
    benchmark.remote(args)
    
    
    
    
    
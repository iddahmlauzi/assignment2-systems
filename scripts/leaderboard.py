import os
import json
import torch
import modal
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.optim import AdamW
from cs336_basics.loss import cross_entropy
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_systems.utils import setup
from cs336_systems.optim import ShardedOptimizer


class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2


cfg = Config()


def timing_worker(rank, world_size, labels, targets):
    import triton
    import triton.testing
    from cs336_basics.model import TransformerLM
    from cs336_systems.fsdp import FSDP
    from cs336_systems.ddp import DDP

    setup(rank, world_size, "cuda")
    device = torch.device(f"cuda:{rank}")

    local_bs = cfg.batch_size // world_size
    labels = labels[rank * local_bs : (rank + 1) * local_bs].to(device)
    targets = targets[rank * local_bs : (rank + 1) * local_bs].to(device)

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.ctx_len,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
    ).to(device=device, dtype=cfg.torch_dtype)

    # model = torch.compile(model)
    model = DDP(model)
    optimizer = ShardedOptimizer(params=model.parameters(), optimizer_cls=AdamW)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = model(labels, checkpoint_k=1)
        loss = cross_entropy(res, targets).sum()
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()


    #timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
    
    for _ in range(1):
        train_step()
        torch.cuda.synchronize()
        dist.barrier()

        
    times = []    
    num_trials = 1
    for i in range(num_trials):
        print(f"Trial: {i}")
        dist.barrier()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        train_step()
        end_event.record()

        torch.cuda.synchronize()
        dist.barrier()

        times.append(start_event.elapsed_time(end_event))

    timing_results = sum(times) / len(times)

    if rank == 0:
        print(timing_results)
        with open("/tmp/result.json", "w") as f:
            json.dump({"timing_results": float(timing_results)}, f)

    dist.destroy_process_group()


@app.function(image=build_image(), 
              volumes=VOLUME_MOUNTS, 
              max_containers=1, 
              gpu="B200:2", 
              timeout=2700)
def test_timing_forward_backward():
    labels, targets = torch.randint(high=cfg.vocab_size, size=(2, cfg.batch_size, cfg.ctx_len))
    mp.spawn(fn=timing_worker, args=(2, labels, targets), nprocs=2, join=True)

    with open("/tmp/result.json") as f:
        return json.load(f)


@app.local_entrypoint()
def main():
    print("1, 2, 3 testing.....")
    print("")
    result = test_timing_forward_backward.remote()
    print(result)
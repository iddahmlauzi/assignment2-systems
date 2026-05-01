# Distributed Data Parallel Training
import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
from jaxtyping import Int
from cs336_systems.utils import setup
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.nvtx as nvtx




class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        
        self.module = module
        self.handles = []
        
        
        def grad_sync_hook(param):
            handle = dist.all_reduce(param.grad, async_op=True)
            self.handles.append(handle)
            
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(grad_sync_hook)
            
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.data /= world_size
    

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        
        self.module = module
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            
        
    def forward(self, input: Int[Tensor, " batch_size sequence_length"]):
        return self.module(input)
    
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        
        all_grads = []
        for param in self.module.parameters():
            if param.grad is None:
                continue
            all_grads.append(param.grad.data)
            
        combined_grad = torch._utils._flatten_dense_tensors(all_grads) 
        dist.all_reduce(combined_grad)
        combined_grad /= world_size
        
        synced_grads = torch._utils._unflatten_dense_tensors(combined_grad, all_grads)
        
        for param, synced_grad in zip(
            [p for p in self.module.parameters() if p.grad is not None], 
            synced_grads
        ):
            param.grad.data.copy_(synced_grad)
            
        
    

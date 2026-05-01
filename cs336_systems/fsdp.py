import einx
import torch
import torch.nn as nn
import torch.distributed as dist
from functools import partial
from cs336_basics.layers import Linear, Embedding


class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        
        self.module = module
        self.compute_dtype = compute_dtype if compute_dtype is not None else torch.float32
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.original_shapes = {} # original shapes for sharded 
        self.shards = {}
        self.sharded_modules = []
        
        self.gather_handles = {}
        self.grad_sync_handles = []
        self.replicated_grad_sync_handles = [] # For RMSNorms 
        
        for name, param in self.module.named_parameters():
            
            # Make sure all params start with the same data
            dist.broadcast(param.data, src=0)
            
            parts = name.rsplit(".", 1)
            mod = dict(self.module.named_modules())[parts[0]] if len(parts) == 2 else self.module
            
            if isinstance(mod, (Linear, Embedding)):
                self.original_shapes[param] = param.shape
                
                # Shard the parameter
                shard = param.data.flatten().chunk(self.world_size)[self.rank].clone()
                param.data = shard
                self.shards[param] = shard # store master weight in original data format
                
                # Store Ordering of the modules
                mod.fsdp_index = len(self.sharded_modules)
                self.sharded_modules.append(mod)
                
                mod.register_forward_pre_hook(self.pre_forward_hook)
                mod.register_forward_hook(self.post_forward_hook)
                param.register_post_accumulate_grad_hook(partial(self.post_backward_hook, mod))
                mod.register_full_backward_pre_hook(self.pre_backward_hook)
                
            else:
                # If it is not (like it is a normal param)
                # We have the all reduce hook for it so it can still be synced
                param.register_post_accumulate_grad_hook(self.grad_sync_hook)
            
            
    def gather_weight_for_layer(self, index):
        if index < 0 or index >= len(self.sharded_modules):
            return
        
        # Don't start a duplicate gather if one is already running
        if index in self.gather_handles:
            return
        
        module = self.sharded_modules[index]
        shard = self.shards[module.weight].to(self.compute_dtype)
        
        full_weight = torch.empty(self.original_shapes[module.weight], device=shard.device, dtype=self.compute_dtype)
        handle = dist.all_gather_into_tensor(full_weight.flatten(), shard, async_op=True)
        
        # Need to store the full weight so we can use it once it is all gathered
        self.gather_handles[index] = (handle, full_weight, shard)
            

    def pre_forward_hook(self, module, args):
        handle, full_weight, _ = self.gather_handles.get(module.fsdp_index, None)
        handle.wait()
        module.weight.data = full_weight # in compute dytpe
        
        if self.needs_full_batch_linear_grad(module):
            module.fsdp_input = args[0].detach()
        
        
    def post_forward_hook(self, module, input, output):
        layer_index = module.fsdp_index
        module.weight.data = self.shards[module.weight] #fp32
        
        # Delete the full weight + the casted and flattened shard
        del self.gather_handles[layer_index]
        
        # Start gathering layer that is twice ahead
        self.gather_weight_for_layer(layer_index + 2)
        
            
    def pre_backward_hook(self, module, grad_output):
        handle, full_weight, _ = self.gather_handles[module.fsdp_index]
        handle.wait()
        module.weight.data = full_weight
        module.weight.grad = None
        
        if self.needs_full_batch_linear_grad(module):
            # derivateive of loss with respect to output
            module.fsdp_grad_output = grad_output[0].detach()


    def needs_full_batch_linear_grad(self, module):
        return self.compute_dtype != torch.float32 and isinstance(module, Linear)


    def full_batch_linear_grad_flat(self, module):
        # Cant figure out how to get fp16 passing without recomputing full gradients
        x = module.fsdp_input
        dy = (module.fsdp_grad_output / self.world_size)

        x_all = torch.empty(
            (x.shape[0] * self.world_size, *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        dy_all = torch.empty(
            (dy.shape[0] * self.world_size, *dy.shape[1:]),
            device=dy.device,
            dtype=dy.dtype,
        )

        dist.all_gather_into_tensor(x_all, x)
        dist.all_gather_into_tensor(dy_all, dy)

        full_grad = einx.dot("... d_out, ... d_in -> d_out d_in", dy_all, x_all)

        del module.fsdp_input
        del module.fsdp_grad_output

        return full_grad.flatten().to(torch.float32)
        

    def post_backward_hook(self, module, param):
        layer_index = module.fsdp_index
        full_grad = param.grad
        
        # Update the weight to be the shard
        param.data = self.shards[param]
        if layer_index in self.gather_handles:
            del self.gather_handles[layer_index]
        self.gather_weight_for_layer(layer_index - 2)
        
        
        if full_grad is not None:
            shard_shape = self.shards[param].shape

            if self.needs_full_batch_linear_grad(module):
                full_grad_flat = self.full_batch_linear_grad_flat(module)
            else:
                full_grad_flat = full_grad.flatten().to(torch.float32)

            shard_gradient = torch.empty(shard_shape, device=module.weight.device, dtype=torch.float32)
            handle = dist.reduce_scatter_tensor(shard_gradient, full_grad_flat, async_op=True)
            
            self.grad_sync_handles.append((handle, module.weight, shard_gradient, full_grad_flat))
            module.weight.grad = None
            
            
            
    def grad_sync_hook(self, param):
        handle = dist.all_reduce(param.grad.data, async_op=True)
        self.replicated_grad_sync_handles.append((handle, param))
           
            
    def forward(self, *inputs, **kwargs):
        # Manually start gathering the first two layers
        self.gather_weight_for_layer(0)
        self.gather_weight_for_layer(1)
        
        output =  self.module(*inputs, **kwargs)
        
        # Get ready for the backward pass
        self.gather_weight_for_layer(len(self.sharded_modules) - 1)
        self.gather_weight_for_layer(len(self.sharded_modules) - 2)
        
        return output
    
        
    def finish_gradient_synchronization(self):
        
        for handle, weight, shard_grad, _ in self.grad_sync_handles:
            handle.wait()
            weight.grad = shard_grad / self.world_size

        for handle, param in self.replicated_grad_sync_handles:
            handle.wait()
            param.grad.data /= self.world_size
                        
        self.grad_sync_handles.clear()
        self.replicated_grad_sync_handles.clear()
        
    def get_full_params_state_dict(self) -> dict[str, torch.Tensor]:
        state_dict = {}
        
        for name, param in self.module.named_parameters():
            if param in self.shards:

                shard = self.shards[param]
                original_shape = self.original_shapes[param]
                full_weight = torch.empty(original_shape, device=shard.device, dtype=shard.dtype)
                dist.all_gather_into_tensor(full_weight.flatten(), shard.flatten())
                state_dict[name] = full_weight
            else:
                # Non-sharded parameters
                state_dict[name] = param.data.clone()
                
        return state_dict
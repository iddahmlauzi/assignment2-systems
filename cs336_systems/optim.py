import torch
import torch.distributed as dist
from typing import Any, Optional
from collections.abc import Callable

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: torch.optim.Optimizer, **kwargs: Any):
        
        defaults = dict(**kwargs)
        self.optimizer_cls = optimizer_cls
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.param_to_rank = {} 
        self.optimizer = None
        self.kwargs = kwargs
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None, **kwargs):
        self.optimizer.step(closure, **kwargs)
        
        # Synchronize with other ranks
        for group in self.param_groups:
            for param in group["params"]:
                dist.broadcast(param.data, src=self.param_to_rank[param])
    
    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        
        all_params = param_group["params"]
        num_params_for_rank = len(all_params) // self.world_size
        
        # I wanna keep track of which rank owns which params for when I broadcast
        rank_params = []
        for i, param in enumerate(all_params):
            owner_rank = i // num_params_for_rank
            # clamp to world_size - 1 in case of remainder so they all go to the very last rank
            owner_rank = min(owner_rank, self.world_size - 1)
            self.param_to_rank[param] = owner_rank
            
            # Keep track of which params to use for this particular rank
            if owner_rank == self.rank:
                rank_params.append(param)

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(rank_params, **self.kwargs)
        else:
            self.optimizer.add_param_group({**param_group, "params": rank_params})
        
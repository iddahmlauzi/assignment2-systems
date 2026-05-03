import torch
import math
import einx
from collections.abc import Callable, Iterable
from typing import Optional
from jaxtyping import Float
from torch import Tensor



def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6) -> None:
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    """
    # How do I calculate the l2 norm
    total = 0
    for p in parameters:
        if p.grad is None:
            continue
        total += torch.sum(p.grad.data ** 2)
    
    l2_norm = math.sqrt(total)
    
    # No clipping needed
    if l2_norm <= max_l2_norm:
        return 
    
    # Clip the Gradients (in-place)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.data.mul_(max_l2_norm / (l2_norm + eps))
    

def get_cosine_lr(it: int,
                  max_learning_rate: float,
                  min_learning_rate: float,
                  warmup_iters: int,
                  cosine_cycle_iters: int):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.
    """
    
    # Warm-Up 
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    
    # Cosine Annealing
    if it <= cosine_cycle_iters:
        cos_term = math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        return min_learning_rate + 0.5 * (1 + cos_term) * (max_learning_rate - min_learning_rate)
    
    return min_learning_rate
    

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas=(0.9, 0.999), lr=1e-3, weight_decay=1e-2, eps=1e-8, device=None, dtype=None):
        if lr < 0:
            raise ValueError(f"Invalid Learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "device": device,
            "dtype": dtype,
        }
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data 
                state = self.state[p]
                
                # Initialize State
                if "t" not in state:
                    state["t"] = 1
                    state["m"] = torch.zeros(grad.shape, device=p.device, dtype=p.dtype)
                    state["v"] = torch.zeros(grad.shape, device=p.device, dtype=p.dtype)
                
                t = state["t"]   
                m = state["m"]
                v = state["v"]
                
                # Update Moment Estimates
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).add_(grad**2, alpha=1-beta2)
                
                # Update parameters
                p.data.sub_(p.data, alpha=weight_decay*lr)          # Weight Decay
                a_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t) # Adjusted learning rate for iteration t
                p.data.sub_(m / (torch.sqrt(v) + eps), alpha=a_t)
                
                state["t"] += 1
                
        return loss
    

# This is what I used in leaderboard but it breaks the normal Adam Tests
# class AdamW(torch.optim.Optimizer):
#     def __init__(self, params, betas=(0.9, 0.999), lr=1e-3, weight_decay=1e-2, eps=1e-8, device=None, dtype=None):
#         if lr < 0:
#             raise ValueError(f"Invalid Learning rate: {lr}")
#         defaults = {
#             "lr": lr,
#             "betas": betas,
#             "eps": eps,
#             "weight_decay": weight_decay,
#             "device": device,
#             "dtype": dtype,
#         }
#         super().__init__(params, defaults)
        
#     def step(self, closure: Optional[Callable] = None):
#         loss = None if closure is None else closure()
#         for group in self.param_groups:
#             lr = group["lr"]
#             beta1, beta2 = group["betas"]
#             eps = group["eps"]
#             weight_decay = group["weight_decay"]
#             device = group["device"]
#             dtype = group["dtype"]
            
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
                
#                 grad = p.grad.data 
#                 state = self.state[p]
                
#                 # Initialize State
#                 if "t" not in state:
#                     state["t"] = 1
#                     state["m"] = torch.zeros(grad.shape, device=device, dtype=dtype)
#                     state["v"] = torch.zeros(grad.shape, device=device, dtype=dtype)
#                     # Okay so we will store the extra mantissa bits for the weight here
#                     # So the weights will technically be in fp32 
#                     state["weight_mantissa"] = torch.zeros(grad.shape, device=device, dtype=torch.uint16)
                
#                 t = state["t"]   
#                 m = state["m"]
#                 v = state["v"]
#                 mantissa = state["weight_mantissa"]
                
#                 # Update Moment Estimates
#                 m.mul_(beta1).add_(grad, alpha=1-beta1)
#                 v.mul_(beta2).add_(grad**2, alpha=1-beta2)
                
#                 # Update parameters
#                 # First, we want to make the data be 32 bits
#                 # What we do is view the underlying data as 16 bits (unsighned)
#                 # Then cast that to uint32 so we have a 32 bit contained
#                 int32_weights = p.data.view(torch.uint16).to(torch.int32)
                
#                 # Then we add the mantissa bits --> Now have a full fp32 weight
#                 int32_weights = (int32_weights << 16) | mantissa.to(torch.int32)
#                 fp32_weights = int32_weights.view(torch.float32)
                
#                 fp32_weights.sub_(fp32_weights, alpha=weight_decay*lr)          # Weight Decay
#                 a_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t) # Adjusted learning rate for iteration t
#                 fp32_weights.sub_(m / (torch.sqrt(v) + eps), alpha=a_t)
                
#                 # Now we want to move back the weights
#                 int32_weights = fp32_weights.view(torch.int32)
#                 state["weight_mantissa"] = int32_weights & 0xFFFF # Low 16 bits
#                 bf16weight = (int32_weights >> 16).to(torch.uint16).view(torch.bfloat16)
#                 p.data = bf16weight
                
#                 state["t"] += 1
                
#         return loss
    
    
# I took these hardcoded coeeficients from the Polar Express Paper: https://arxiv.org/pdf/2505.16932 
# I also made them smaller for stability (by dividing by 1.01 *)
polar_express_coefficients = [
    (8.205160414005574, -22.90193498705605, 16.460724910180314),
    (4.066395159942775, -2.8611540867551426, 0.5183995226694741),
    (3.9095949044379155, -2.823351735039516, 0.5250369769390025),
    (3.2855640171986153, -2.415301959635945, 0.48529406552790866),
    (2.277873287083977, -1.619821765265441, 0.39848078704168355)
]

def polar_express(G: Float[Tensor, "..."], eps=1e-2):    
    # Reduces memory cost when calculating X^TX
    rows, cols = G.shape[-2], G.shape[-1]
    if rows > cols:
        G = G.transpose(-2, -1)
    
    G = G / (G.norm(p='fro', dim=(-2, -1), keepdim=True) * 1.01 + eps)
    n = G.shape[-2] # This will be the dimension of the square matrix when we say X^T
    I = torch.eye(n, dtype=G.dtype, device=G.device)  # This is a square matrix of n * n
    for a, b, c in polar_express_coefficients:
        S = einx.dot("... i [j], ... k [j] -> ... i k", G, G)
        intermediate = a * I + S @ (b * I + c * S)
        G = intermediate @ G
    
    if rows > cols:
        G = G.transpose(-2, -1)
        
    return G
    

# Used the psuedocode here: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html 
# Also look into Polar Express 
# And NorMuon
# https://www.youtube.com/watch?v=-Cto66pAUXQ 
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=1e-1, nesterov=True, device=None, dtype=None):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "device": device,
            "dtype": dtype
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            device = group["device"]
            dtype = group["dtype"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data 
                state = self.state[p]
                
                # Initialize State
                if "m" not in state:
                    state["m"] = torch.zeros(grad.shape, device=device, dtype=dtype)
                    # Okay so we will store the extra mantissa bits for the weight here
                    # So the weights will technically be in fp32 
                    state["weight_mantissa"] = torch.zeros(grad.shape, device=device, dtype=torch.uint16)
                  
                m = state["m"]
                mantissa = state["weight_mantissa"]
                
                # Update the gradient buffer
                m.mul_(momentum).add_(grad)
                
                if nesterov:
                    m.mul_(momentum).add_(grad)
                    
                # Then we want to orthogonalize the gradient
                O_t = polar_express(grad)
                
                
                # Update parameters
                # First, we want to make the data be 32 bits
                # What we do is view the underlying data as 16 bits (unsighned)
                # Then cast that to uint32 so we have a 32 bit contained
                int32_weights = p.data.view(torch.uint16).to(torch.int32)
                
                # Then we add the mantissa bits --> Now have a full fp32 weight
                int32_weights = (int32_weights << 16) | mantissa.to(torch.int32)
                fp32_weights = int32_weights.view(torch.float32)
                fp32_weights.sub_(fp32_weights, alpha=weight_decay*lr)   # Weight Decay
                
                
                A, B = p.grad.shape
                a_t = 0.2 * lr * math.sqrt(max(A, B)) # Adjusted learning rate for iteration t --> Match the RMS of ADAM
                fp32_weights.sub_(O_t, alpha=a_t)
                
                # Now we want to move back the weights
                int32_weights = fp32_weights.view(torch.int32)
                state["weight_mantissa"] = int32_weights & 0xFFFF # Low 16 bits
                bf16weight = (int32_weights >> 16).to(torch.uint16).view(torch.bfloat16)
                p.data = bf16weight

                
        return loss
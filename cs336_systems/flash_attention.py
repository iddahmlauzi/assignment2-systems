import einx
import torch 
import math
from jaxtyping import Float
from torch import Tensor

Q_TILE_SIZE = 32
K_TILE_SIZE = 32

class FlashAttentionPytorch(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,
                Q: Float[Tensor, " ... queries d_k"],
                K: Float[Tensor, " ... keys d_k"],
                V: Float[Tensor, " ... keys d_v"],
                is_causal: bool=False):
        
        
        N, d_k = Q.shape[-2], Q.shape[-1]
        d_v = V.shape[-1] # In practice, this should be the same as d_k
        
        device = Q.device
        O = torch.zeros((*Q.shape[:-1], d_v), device=device)
        L = torch.zeros(Q.shape[:-1], device=device)                            # softmax denominator
        M = torch.full(Q.shape[:-1], fill_value=-float("inf"), device=device)   # row maximum
        
        for i in range(math.ceil(N / Q_TILE_SIZE)):
            # Load block from Q
            Q_i = Q[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE, :]    # (..., q_tile_size, d_k)
            o_prev = O[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE, :] # (..., q_tile_size, d_v)
            l_prev = L[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE]    # (..., q_tile_size)
            m_prev = M[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE]    # (..., q_tile_size)
            
     
            for j in range(math.ceil(N / K_TILE_SIZE)):
                K_j = K[..., j * K_TILE_SIZE: (j + 1) * K_TILE_SIZE, :] # (..., k_tile_size, d_k)
                V_j = V[..., j * K_TILE_SIZE: (j + 1) * K_TILE_SIZE, :] # (..., k_tile_size, d_v)
                
                # Pre-Softmax Attention Scores
                S = einx.dot("... q_tile_size [dk], ... k_tile_size [dk] -> ... q_tile_size k_tile_size", Q_i, K_j) / math.sqrt(d_k)
                rowmax, _ = torch.max(S, dim=-1) # (..., q_tile_size)
                m_new = torch.maximum(m_prev, rowmax) # (..., q_tile_size)
                
                P = torch.exp(S - m_new.unsqueeze(-1)) # (..., q_tile_size, k_tile_size)
                correction_factor =  torch.exp(m_prev - m_new) # (..., q_tile_size)
                l_prev = correction_factor * l_prev + torch.sum(P, dim=-1) # Rescale the running denominator based on new maximum value
                o_prev = correction_factor.unsqueeze(-1) * o_prev + P @ V_j # (..., q_tile_size, d_v)
                
                m_prev = m_new
                
            O[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE, :] = o_prev
            L[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE] = l_prev
            M[..., i * Q_TILE_SIZE: (i + 1) * Q_TILE_SIZE] = m_prev
            
                
        # Normalize the final output
        O = O / L.unsqueeze(-1)
        L = M + torch.log(L)
        
        ctx.save_for_backward(L, Q, K, V, O)
        
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("We have not implemented this yet")
    
    
    
    
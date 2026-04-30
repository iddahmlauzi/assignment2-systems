import einx
import torch 
import math
import triton
import triton.language as tl
from jaxtyping import Float
from torch import Tensor

PYTORCH_Q_TILE_SIZE = 32
PYTORCH_K_TILE_SIZE = 32

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
    ):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # Initialize buffers to write to
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    M = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    
    q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        
        S = tl.dot(Q, tl.trans(K)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        
        if is_causal:
            k_indices = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_indices[:, None] < k_indices[None, :]  # (Q_TILE_SIZE, K_TILE_SIZE)
            S = tl.where(mask, S + -1e6, S)
        rowmax = tl.max(S, axis=1)                          # (Q_TILE_SIZE,)
        m_new = tl.maximum(M, rowmax)                       # (Q_TILE_SIZE,)
        
        P = tl.exp(S - m_new[:, None])                      # (Q_TILE_SIZE, K_TILE_SIZE)
        correction_factor = tl.exp(M - m_new)               # (Q_TILE_SIZE,)
        L = correction_factor * L + tl.sum(P, axis=1)       # (Q_TILE_SIZE,)
        O = correction_factor[:, None] * O
        O = tl.dot(P.to(V.dtype), V, acc=O)                 # (Q_TILE_SIZE, D)
        
        M = m_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    
    # Normalize the final output
    O = O / L[:, None]
    L = M + tl.log(L)
    
    # Write the final outputs
    O_dtype = O_block_ptr.type.element_ty
    tl.store(O_block_ptr, O.to(O_dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))
    
    
def flash_bwd_pytorch(
            Q: Float[Tensor, " ... queries d_k"],
            K: Float[Tensor, " ... keys d_k"],
            V: Float[Tensor, " ... keys d_v"],
            O: Float[Tensor, " ... queries d_v"],
            dO: Float[Tensor, " ... queries d_v"],
            L: Float[Tensor, " ... queries"],
            is_causal: bool=False,
            ):
    _, n_queries, d = Q.shape
    n_keys = K.shape[-2]
    scale = 1 / math.sqrt(d)
    
    D = torch.sum(O * dO, dim=-1) # (batch, queries)
    S = einx.dot("... queries [d_k], ... keys [d_k] -> ... queries keys", Q, K) * scale
    if is_causal:
        mask = torch.tril(torch.ones(n_queries, n_keys, device=Q.device, dtype=torch.bool))
        S.masked_fill_(~mask, -float("inf"))
    
    P = torch.exp(S - L.unsqueeze(-1)).to(Q.dtype) # (batch, queries, keys)
    
    dV = einx.dot("... [queries] keys, ... [queries] d_v -> ... keys d_v", P, dO)
    dP = einx.dot("... queries [d_v], ... keys [d_v] -> ... queries keys", dO, V)
    dS = P * (dP - D.unsqueeze(-1)) # (batch, queries, keys)
    dQ = einx.dot("... queries [keys],  ... [keys] d_k -> ... queries d_k", dS, K) * scale
    dK = einx.dot("... [queries] keys, ... [queries] d_k ->  ... keys d_k", dS, Q) * scale
    
    return dQ, dK, dV
    
    
    
class FlashAttentionTriton(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,
            Q: Float[Tensor, " ... queries d_k"],
            K: Float[Tensor, " ... keys d_k"],
            V: Float[Tensor, " ... keys d_v"],
            is_causal: bool=False,
            ):

        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        scale = 1 / math.sqrt(d)
        
        device = Q.device
        
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.D = d
        ctx.is_causal = is_causal
        
        O = torch.empty((batch_size, n_queries, d), device=device, dtype=Q.dtype)
        L = torch.empty((batch_size, n_queries), device=device, dtype=torch.float32)  
        
        flash_fwd_kernel[(triton.cdiv(n_queries, ctx.Q_TILE_SIZE), batch_size)](
            Q, K, V, 
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            ctx.D,
            ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE,
            ctx.is_causal
        )
    
        ctx.save_for_backward(Q, K, V, L, O)
        
        return O
    
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV = flash_bwd_pytorch(
            Q, K, V, O, grad_out, L, ctx.is_causal
        )
        return dQ, dK, dV, None
    

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
        
        for i in range(math.ceil(N / PYTORCH_Q_TILE_SIZE)):
            # Load block from Q
            Q_i = Q[..., i * PYTORCH_Q_TILE_SIZE: (i + 1) * PYTORCH_Q_TILE_SIZE, :]    # (..., q_tile_size, d_k)
            o_prev = O[..., i * PYTORCH_Q_TILE_SIZE: (i + 1) * PYTORCH_Q_TILE_SIZE, :] # (..., q_tile_size, d_v)
            l_prev = L[..., i * PYTORCH_Q_TILE_SIZE: (i + 1) * PYTORCH_Q_TILE_SIZE]    # (..., q_tile_size)
            m_prev = M[..., i * PYTORCH_Q_TILE_SIZE: (i + 1) * PYTORCH_Q_TILE_SIZE]    # (..., q_tile_size)
            
     
            for j in range(math.ceil(N / PYTORCH_K_TILE_SIZE)):
                K_j = K[..., j * PYTORCH_K_TILE_SIZE: (j + 1) * PYTORCH_K_TILE_SIZE, :] # (..., k_tile_size, d_k)
                V_j = V[..., j * PYTORCH_K_TILE_SIZE: (j + 1) * PYTORCH_K_TILE_SIZE, :] # (..., k_tile_size, d_v)
                
                # Pre-Softmax Attention Scores
                S = einx.dot("... q_tile_size [dk], ... k_tile_size [dk] -> ... q_tile_size k_tile_size", Q_i, K_j) / math.sqrt(d_k)
                rowmax, _ = torch.max(S, dim=-1) # (..., q_tile_size)
                m_new = torch.maximum(m_prev, rowmax) # (..., q_tile_size)
                
                P = torch.exp(S - m_new.unsqueeze(-1)) # (..., q_tile_size, k_tile_size)
                correction_factor =  torch.exp(m_prev - m_new) # (..., q_tile_size)
                l_prev = correction_factor * l_prev + torch.sum(P, dim=-1) # Rescale the running denominator based on new maximum value
                o_prev = correction_factor.unsqueeze(-1) * o_prev + P @ V_j # (..., q_tile_size, d_v)
                
                m_prev = m_new
                
            O[..., i * PYTORCH_Q_TILE_SIZE: (i + 1) * PYTORCH_Q_TILE_SIZE, :] = o_prev
            L[..., i * PYTORCH_Q_TILE_SIZE: (i + 1) * PYTORCH_Q_TILE_SIZE] = l_prev
            M[..., i * PYTORCH_Q_TILE_SIZE: (i + 1) * PYTORCH_Q_TILE_SIZE] = m_prev
            
                
        # Normalize the final output
        O = O / L.unsqueeze(-1)
        L = M + torch.log(L)
        
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        
        return O
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV = flash_bwd_pytorch(
            Q, K, V, O, grad_out, L, ctx.is_causal
        )
        return dQ, dK, dV, None
    
    

    
    
    
    
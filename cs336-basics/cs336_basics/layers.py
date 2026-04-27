"""
Core layers and operations for the Transformer LM.
- Math helpers: silu, softmax, scaled dot-product attention
- Custom modules: Linear, Embedding, RMSNorm, RoPE
"""

import torch
import torch.nn as nn
import math
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch.cuda.nvtx as nvtx



def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Swish Activation function"""
    return x * torch.sigmoid(x)

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.
    """
    # Numerical stability. Make the max value in each become 0
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)

@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"],
                       K: Float[Tensor, " ... keys d_k"],
                       V: Float[Tensor, " ... keys d_v"],
                       mask: Bool[Tensor, " ... queries keys"] | None = None,
                       cap_logits=False
                       ) -> Float[Tensor, " ... queries d_v"]:
    
    d_k = Q.shape[-1]
    
    with nvtx.range("computing attention scores"):
        attn = einx.dot("... queries [d_k], ... keys [d_k] -> ... queries keys", Q, K) / math.sqrt(d_k)
        
    with nvtx.range("causal masking"):
        # The mask is True where we want to attend --> so we need to invert it so it is True where we do not want
        attn.masked_fill_(~mask, -float("inf"))
        
        # Add logit softcapping: https://arxiv.org/pdf/2408.00118 
        if cap_logits:
            # Probably not a good idea to hardcode this but ehh
            attn = 50 * torch.tanh(attn / 50)
        
    with nvtx.range("computing softmax"):
        softmax_value = softmax(attn, dim=-1)
        
    with nvtx.range("final matmul"):
        return einx.dot("... queries keys, ... keys d_v -> ... queries d_v", softmax_value, V)


class Linear(nn.Module):
    """Linear tranformation module"""
    def __init__(self, in_features: int, out_features: int, zero_init=False, device=None, dtype=None):
        super().__init__()
        
        std = math.sqrt(2 / (in_features + out_features))
        # row-major ordering
        weight_tensor = torch.empty(out_features, in_features, dtype=dtype, device=device)
        
        # Zero-Init of the projection layers
        if zero_init:
            nn.init.zeros_(weight_tensor)
        else:
            nn.init.trunc_normal_(weight_tensor, a=-3*std, b=3*std)
        self.weight = nn.Parameter(weight_tensor)
        
    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        """Calculates xW^T"""
        return einx.dot("d_out d_in, ... d_in -> ... d_out", self.weight, x)
    
    
class Embedding(nn.Module):
    """Embedding Layer"""
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        
        embedding_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(embedding_tensor, std=1, a=-3, b=3)
        self.weight = nn.Parameter(embedding_tensor)
    
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return einx.get_at("[vocab_size] d_model, ... -> ... d_model", self.weight, token_ids)
    
    
class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        # Are we making this initialized to 1 so that initially the scale factor is basically a no-op?
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Upcast to prevent overflow when squaring
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = einx.mean("... [d_model] -> ... 1", x * x + self.eps)
        result = x / torch.sqrt(rms) * self.weight
        
        # Return the result in the original dtype
        return result.to(in_dtype)
    
    
class RotaryPositionalEmbedding(nn.Module):
    """Implements RoPE embeddings"""
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        # Get indices [0, 1, .... max_seq_len]
        indices = torch.arange(start=0, end=max_seq_len, dtype=dtype, device=device)
        # Get k values [1, 2, .... d/2]
        k = torch.arange(start=1, end=d_k/2 + 1, dtype=dtype, device=device)
        # Denominator is theta^(2k - 2)/d
        denominator = theta ** ((2 * k - 2) / d_k)
        # All possible theta values we could take on
        thetas = einx.divide("max_seq_len, ... -> max_seq_len ...", indices, denominator)
        self.register_buffer("cos", torch.cos(thetas), persistent=False)
        self.register_buffer("sin", torch.sin(thetas), persistent=False)

        
    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_k"]:
        # Get the pairs in the data
        x = einx.id("... sequence_length (num_pairs d) -> ... sequence_length num_pairs d", x, d=2)

        # Get cos and sin values for the token positions
        cos = einx.get_at("[max_seq_len] d, ... sequence_length -> ... sequence_length d", self.cos, token_positions)
        sin = einx.get_at("[max_seq_len] d, ... sequence_length -> ... sequence_length d", self.sin, token_positions)

        rotated_pair_x = cos * x[..., 0] - sin * x[..., 1]
        rotated_pair_y = sin * x[..., 0] + cos * x[..., 1]
        
        # Put them together
        rotated = einx.id("... sequence_length d, ... sequence_length d -> ... sequence_length d (1 + 1)", rotated_pair_x, rotated_pair_y)
        
        return einx.id("... sequence_length d p -> ... sequence_length (d p)", rotated)
    

    

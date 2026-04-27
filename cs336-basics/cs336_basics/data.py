import torch
import jaxtyping
from torch import Tensor
import numpy.typing as npt
import numpy as np


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    """
    
    # Sample indices from [0 up to n - m]
    n = dataset.size  # (batch_size, context_length)
    start_indices = np.random.randint(low=0, high= n - context_length, size=batch_size)  # (batch_size, )
    offsets = np.arange(start=0, stop=context_length) # (context_len,)
    
    start_indices = np.expand_dims(start_indices, 1) # (batch_size, 1)
    offsets = np.expand_dims(offsets, 0) # (1, context_len)
    all_indices = start_indices + offsets # (batch_size, context_len)
    
    inputs, targets = dataset[all_indices], dataset[all_indices + 1]
    inputs =  torch.as_tensor(inputs, device=device, dtype=torch.long)
    targets = torch.as_tensor(targets, device=device, dtype=torch.long)
    
    return inputs, targets
    
    
    
    
    
    

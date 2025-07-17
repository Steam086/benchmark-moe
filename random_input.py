import torch
import random
import numpy as np
import os

# TODO use the same seed 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

    # optional may affect performance
    torch.use_deterministic_algorithms(True)
# set random seed for reproducibility
set_seed(42)


def prepare_weights(num_experts:int,
                    intermediate_size:int,
                    hidden_size:int,
                    init_dtype:torch.dtype=torch.float16) -> tuple[torch.Tensor, torch.Tensor]:

    w1 = torch.randn(
        num_experts, intermediate_size, hidden_size, dtype=init_dtype
    )
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size // 2, dtype=init_dtype
    )
    return w1, w2


def prepare_inputs(
    num_iters,
    num_tokens,
    hidden_size,
    num_experts,
    dtype: torch.dtype=torch.float16
) -> tuple[torch.Tensor, torch.Tensor]:
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return x, gating_output




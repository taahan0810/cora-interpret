import numpy as np
import torch

def set_seeds(seed_value=123):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
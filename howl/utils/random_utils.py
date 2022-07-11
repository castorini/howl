import random

import numpy as np
import torch


def set_random_seed(seed: int = 0):
    """Set random seed

    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

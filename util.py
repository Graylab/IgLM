import random
import numpy as np
import transformers
import torch


def set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    transformers.set_seed(0)

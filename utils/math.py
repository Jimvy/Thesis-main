r"""
Polymorphic maths :O
Because torch's math functions suck...
"""

import torch
import numpy


def sqrt(x):
    if isinstance(x, torch.Tensor):
        return  torch.sqrt(x)
    else:
        return numpy.sqrt(x)  # Can catch most of it

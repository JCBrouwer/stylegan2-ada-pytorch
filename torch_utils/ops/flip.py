import torch


def flip(x, dims):
    return x[
        tuple(slice(None, None) if i in dims else torch.arange(x.size(i) - 1, -1, -1).long() for i in range(x.dim()))
    ]

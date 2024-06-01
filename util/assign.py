import torch.nn

auto = False


def assign(left, right):
    if left.shape != right.shape:
        if left.shape == right.T.shape and auto:
            right = right.T
        else:
            raise ValueError(f"Shape mismatch - Left: {left.shape} Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
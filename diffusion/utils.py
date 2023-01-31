import torch


def extract(
    a: torch.Tensor,
    t: torch.Tensor
):
    batch_size = t.shape[0]
    out = torch.gather(a, -1, t)
    return out.reshape(batch_size, 1, 1, 1)

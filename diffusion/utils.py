from torch import Tensor
from typing import Tuple


def extract(
    arr: Tensor,
    timesteps: Tensor,
    broadcast_shape: Tuple[int]
):
    broadcast_shape = (len(timesteps), *broadcast_shape)
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

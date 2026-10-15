from collections.abc import Iterable

import torch
import torch.nn as nn
from einops import reduce
from jaxtyping import Float, Int
from torch import Tensor, sigmoid


def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Apply SiLU elementwise, preserving the input shape."""
    return x * sigmoid(x)


def softmax(x: Float[Tensor, " ..."], dim: int = -1) -> Float[Tensor, " ..."]:
    """Normalize scores along ``dim`` while preserving the input shape.

    Subtracting each maximum before exponentiation avoids overflow for large
    finite scores. Each normalized slice must contain a finite score; a slice
    containing only negative infinity produces NaNs.
    """
    moved = x.movedim(dim, -1)
    shifted = moved - reduce(moved, "... feature -> ... 1", "max")
    exp_x = torch.exp(shifted)
    result = exp_x / reduce(exp_x, "... feature -> ... 1", "sum")
    return result.movedim(-1, dim)


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    shifted = inputs - reduce(inputs, "... vocab_size -> ... 1", "max")
    logits_sum = reduce(torch.exp(shifted), "... vocab_size -> ... 1", "sum")
    target_logits = shifted.gather(-1, targets.unsqueeze(-1))
    loss = torch.log(logits_sum) - target_logits
    return loss.mean()


def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    l2_norm = 0.0
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    for grad in grads:
        l2_norm += grad.square().sum()
    l2_norm = l2_norm.sqrt()
    if l2_norm >= max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for grad in grads:
            grad.mul_(scale)

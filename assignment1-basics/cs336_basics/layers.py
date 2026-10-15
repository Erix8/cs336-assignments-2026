import torch
import torch.nn as nn
from einops import einsum, reduce
from torch import Tensor

from cs336_basics.nn_utils import silu


class Linear(nn.Module):
    """Bias-free projection with weights stored as ``(out_features, in_features)``.

    Truncated-normal initialization limits extreme initial weights. Only the
    final input dimension is projected; all leading dimensions are preserved.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_in, d_out = in_features, out_features
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0, std, -3 * std, 3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Learnable token lookup with truncated-normal initialization.

    Integer IDs of shape ``(...)`` produce embeddings of shape
    ``(..., embedding_dim)``. IDs must be in ``[0, num_embeddings)``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Normalize the final dimension using a learned per-channel scale.

    Inputs have shape ``(..., d_model)``. Float32 computation reduces error in
    the squared-value reduction; outputs retain the input shape and dtype.
    ``eps`` stabilizes normalization near zero.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        return (x / rms * self.weight).to(in_dtype)


class FeedForward(nn.Module):
    """Apply SwiGLU independently at each token, preserving ``(..., d_model)``.

    When omitted, ``d_ff`` is chosen near ``8 * d_model / 3``, rounded to a
    multiple of 64 to balance parameter count and matrix-operation efficiency.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = round(8 * d_model / 192) * 64 if d_ff is None else d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

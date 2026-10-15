import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from cs336_basics.layers import Linear
from cs336_basics.nn_utils import softmax


class RoPE(nn.Module):
    """Rotate adjacent feature pairs using fixed position-dependent angles.

    ``theta`` is the frequency base, and ``d_k`` must be even. The initial
    float32 sine/cosine tables cover positions ``[0, max_seq_len)``. They are
    nonpersistent buffers because they can be regenerated from configuration.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inv_freq = theta ** (-exponents)
        angles = einsum(positions, inv_freq, "seq, pair -> seq pair")
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Rotate ``(..., seq, d_k)`` inputs, preserving their shape and dtype.

        Integer positions index the cached tables. The resulting coefficients
        must broadcast to ``(..., seq, d_k // 2)``; callers insert singleton
        dimensions when sharing positions across heads. Rotation arithmetic
        uses float32 before casting back to the input dtype.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        pairs = rearrange(x, "... seq (pair component) -> ... seq pair component", component=2)
        a, b = pairs[..., 0], pairs[..., 1]
        cos, sin = self.cos[token_positions], self.sin[token_positions]
        a_rotated, b_rotated = a * cos - b * sin, a * sin + b * cos
        rotated_pairs = torch.stack((a_rotated, b_rotated), dim=-1)
        return rearrange(rotated_pairs, "... seq pair component -> ... seq (pair component)").to(in_dtype)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
):
    """Return attention outputs of shape ``(..., queries, d_v)``.

    Q and K share a feature dimension; K and V share a sequence length.
    Leading dimensions represent independent batches or heads. The optional
    boolean mask must broadcast to the scores, with True allowing attention.
    Each query must have at least one unmasked key with a finite score.
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probalibities = softmax(scores / (d_k**0.5), dim=-1)
    return einsum(probalibities, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(nn.Module):
    """Causal self-attention with optional RoPE applied to each head's Q and K.

    ``num_heads`` must be positive and divide ``d_model``. An injected RoPE
    module must use ``d_k = d_model // num_heads`` and can be shared by layers.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads must be positive and divide d_model")
        self.rope = rope
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.o_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Map ``(..., seq, d_model)`` to the same shape using causal attention.

        Optional integer positions have shape ``(..., seq)`` without a head
        dimension and are shared across heads. With RoPE enabled, omitted
        positions default to ``0, ..., seq - 1``. Causality follows sequence
        order rather than the supplied position values.
        """
        seq_len = x.shape[-2]
        Q = rearrange(self.q_proj(x), "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (head d_v) -> ... head seq d_v", head=self.num_heads)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        casual_attention = scaled_dot_product_attention(Q, K, V, mask)
        merged = rearrange(casual_attention, "... head seq d_v -> ... seq (head d_v)")
        return self.o_proj(merged)

"""Transformer language-model components for CS336 assignment 1."""

import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor
from torch import sigmoid
from einops import einsum, reduce, rearrange


def silu(x: Tensor) -> Tensor:
    """Apply SiLU elementwise, preserving the input shape."""
    return x * sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
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


def swiglu(x: Tensor, w1: Linear, w2: Linear, w3: Linear) -> Tensor:
    """Apply a SiLU-gated hidden projection followed by an output projection.

    w1 and w3 map the input to the same hidden width; w2 maps their gated
    product to the output width.
    """
    return w2(silu(w1(x)) * w3(x))


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
        return swiglu(x, self.w1, self.w2, self.w3)


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


class TransformerBlock(nn.Module):
    """Pre-norm causal attention and SwiGLU, each with a residual connection.

    An injected ``rope`` takes precedence over ``theta``. Otherwise, a supplied
    ``theta`` creates a local RoPE cache using ``max_seq_len``; omitting both
    disables RoPE. Inputs and outputs have shape ``(..., seq, d_model)``.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float | None = None,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.rope = rope
        if self.rope is None and theta is not None:
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype, rope=self.rope)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        y = x + self.attn(self.ln1(x))
        return y + self.ffn(self.ln2(y))


class Transformer(nn.Module):
    """Causal language model with independent blocks sharing one RoPE cache.

    ``context_length`` sets the cached position range. Token embeddings and
    the vocabulary output projection have independent trainable weights.
    A final RMSNorm precedes the output projection.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                    rope=self.rope,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Map integer token IDs ``(..., seq)`` to logits ``(..., seq, vocab_size)``.

        Sequence length must not exceed ``context_length``. Logits remain
        unnormalized so training losses can operate directly on scores.
        """
        y = self.token_embeddings(x)
        for layer in self.layers:
            y = layer(y)
        z = self.ln_final(y)
        return self.lm_head(z)

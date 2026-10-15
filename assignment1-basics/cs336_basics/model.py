"""Transformer language-model components for CS336 assignment 1."""

import torch
import torch.nn as nn
from torch import Tensor

from cs336_basics.attention import MultiHeadSelfAttention, RoPE
from cs336_basics.layers import Embedding, FeedForward, Linear, RMSNorm


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

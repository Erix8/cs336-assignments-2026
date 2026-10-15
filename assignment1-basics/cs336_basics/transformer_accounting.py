"""Print answers to transformer_accounting using assignment-architecture formulas.

Count dense forward matrix multiplies for one sequence, including all masked
attention entries. Exclude elementwise operations, buffers, and activations.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_layers: int
    d_model: int
    num_heads: int
    vocab_size: int = 50_257
    context_length: int = 1_024

    @property
    def d_ff(self) -> int:
        return round((8 * self.d_model / 3) / 64) * 64


def parameter_counts(config: ModelConfig) -> dict[str, int]:
    layers, d, f, vocab = config.num_layers, config.d_model, config.d_ff, config.vocab_size
    # Embeddings and the vocabulary projection have independent weights.
    return {
        "Token embedding": vocab * d,
        "QKV projections": layers * 3 * d * d,
        "Output projections": layers * d * d,
        "SwiGLU FFN": layers * 3 * d * f,
        "Block RMSNorms": layers * 2 * d,
        "Final RMSNorm": d,
        "LM head": vocab * d,
    }


def forward_flops(config: ModelConfig) -> dict[str, int]:
    layers, t, d, f = config.num_layers, config.context_length, config.d_model, config.d_ff
    # Summing heads cancels their count because num_heads * d_k = d_model.
    return {
        "QKV projections": layers * 6 * t * d * d,
        "Output projections": layers * 2 * t * d * d,
        "QK^T": layers * 2 * t * t * d,
        "Attention times V": layers * 2 * t * t * d,
        "SwiGLU FFN": layers * 6 * t * d * f,
        "LM head": 2 * t * d * config.vocab_size,
    }


def print_flops(config: ModelConfig) -> None:
    counts = forward_flops(config)
    total = sum(counts.values())
    print(
        f"{config.name}: L={config.num_layers}, d={config.d_model}, "
        f"h={config.num_heads}, f={config.d_ff}, T={config.context_length}"
    )
    print(f"{'Component':<24} {'FLOPs':>20} {'Share':>9}")
    for name, count in counts.items():
        print(f"{name:<24} {count:>20,} {count / total:>8.2%}")
    print(f"{'Total':<24} {total:>20,} {1:>8.2%}")
    print(f"Total: {total / 1e12:.9f} TFLOPs\n")


def main() -> None:
    small = ModelConfig("GPT-2 small", 12, 768, 12)
    medium = ModelConfig("GPT-2 medium", 24, 1_024, 16)
    large = ModelConfig("GPT-2 large", 36, 1_280, 20)
    xl = ModelConfig("GPT-2 XL", 48, 1_600, 25)

    print("Assignment architecture: bias-free, SwiGLU, RMSNorm, RoPE, untied embeddings.")
    print("Batch size = 1. FLOPs count dense matrix multiplies only (2*m*n*p).\n")

    print("(a) Trainable parameters and FP32 parameter memory")
    counts = parameter_counts(xl)
    for name, count in counts.items():
        print(f"{name:<24} {count:>20,}")
    total_parameters = sum(counts.values())
    memory_bytes = 4 * total_parameters
    print(f"Total parameters: {total_parameters:,}")
    print(f"FP32 memory: {memory_bytes:,} bytes = {memory_bytes / 1e9:.6f} GB = {memory_bytes / 2**30:.6f} GiB\n")

    print("(b) Forward matrix multiplies and FLOPs")
    print("Per layer, summed over heads:")
    print("Q/K/V: 3 * (T,d) @ (d,d) -> 6*T*d^2")
    print("QK^T: h * (T,d/h) @ (d/h,T) -> 2*T^2*d")
    print("Attention times V: h * (T,T) @ (T,d/h) -> 2*T^2*d")
    print("Output: (T,d) @ (d,d) -> 2*T*d^2")
    print("FFN: 2 * (T,d) @ (d,f) + (T,f) @ (f,d) -> 6*T*d*f")
    print("Final LM head: (T,d) @ (d,V) -> 2*T*d*V")
    print("Total: L*(8*T*d^2 + 4*T^2*d + 6*T*d*f) + 2*T*d*V\n")
    print_flops(xl)

    print("(c) Largest FLOPs contributors")
    xl_flops = forward_flops(xl)
    xl_total = sum(xl_flops.values())
    ranked = sorted(xl_flops.items(), key=lambda item: item[1], reverse=True)
    for name, count in ranked:
        print(f"{name}: {count / xl_total:.2%}")
    attention = xl_flops["QK^T"] + xl_flops["Attention times V"]
    print(f"Combined attention matrix multiplies: {attention / xl_total:.2%}\n")

    print("(d) Model-size comparison")
    for config in (small, medium, large, xl):
        print_flops(config)
    print("At fixed T and V, projections and FFN scale as L*d^2, attention as L*d,")
    print("and LM head as d. Projections/FFN gain share; attention/LM head lose share.\n")

    print("(e) GPT-2 XL with context length 16,384")
    long_xl = replace(xl, context_length=16_384)
    print_flops(long_xl)
    long_flops = forward_flops(long_xl)
    long_total = sum(long_flops.values())
    long_attention = long_flops["QK^T"] + long_flops["Attention times V"]
    print(f"Total FLOPs increase: {long_total / xl_total:.6f}x")
    print("Projections, FFN, and LM head increase 16x; attention matmuls increase 256x.")
    print(f"Combined attention share: {attention / xl_total:.2%} -> {long_attention / long_total:.2%}")
    print(f"Trainable parameters remain {total_parameters:,}.")


if __name__ == "__main__":
    main()

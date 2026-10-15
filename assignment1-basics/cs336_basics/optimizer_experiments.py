import argparse
from dataclasses import dataclass
from fractions import Fraction

import torch

from cs336_basics.optimizer import SGD


def learning_rate_tuning():
    lrs = [1.0, 1e1, 1e2, 1e3]
    generator = torch.Generator().manual_seed(42)
    initial_weights = 5 * torch.randn((10, 10), generator=generator)
    histories = {}
    for lr in lrs:
        print(f"Learning Rate: {lr}")
        # Match initial conditions without sharing mutable weights or optimizer state.
        weights = torch.nn.Parameter(initial_weights.clone())
        opt = SGD([weights], lr=lr)
        losses = [weights.detach().square().mean().item()]
        print(f"Step 0: {losses[0]:.8e}")
        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                losses.append(weights.square().mean().item())
            print(f"Step {t + 1}: {losses[-1]:.8e}")
        histories[lr] = losses
        print()
    return histories


@dataclass(frozen=True)
class AccountingConfig:
    vocab_size: int = 50_257
    context_length: int = 1_024
    num_layers: int = 48
    d_model: int = 1_600
    num_heads: int = 25
    d_ff: int | None = None

    @property
    def ff_width(self) -> Fraction:
        # Section 4.3 specifies an exact ratio, unlike the rounded earlier example.
        return Fraction(8 * self.d_model, 3) if self.d_ff is None else Fraction(self.d_ff)


def adamw_parameter_counts(config: AccountingConfig) -> dict[str, Fraction]:
    d, f, layers = config.d_model, config.ff_width, config.num_layers
    return {
        "Untied token embedding and LM head": Fraction(2 * config.vocab_size * d),
        "Attention projections": Fraction(layers * 4 * d**2),
        "SwiGLU weights": layers * 3 * d * f,
        "RMSNorm weights": Fraction((2 * layers + 1) * d),
    }


def activation_counts_per_sequence(config: AccountingConfig) -> dict[str, Fraction]:
    """Estimate retained FP32 outputs without checkpointing or fused attention.

    Count each listed operator's output once, with one vocabulary-sized
    log-probability tensor for cross-entropy. Exclude residual additions,
    RMSNorm statistics, RoPE, masks, and implementation-specific temporaries.
    This is an accounting convention, not an allocator peak measurement.
    """
    t, d, f = config.context_length, config.d_model, config.ff_width
    layers, heads, vocab = config.num_layers, config.num_heads, config.vocab_size
    return {
        "Block RMSNorm outputs": Fraction(layers * 2 * t * d),
        "QKV projection outputs": Fraction(layers * 3 * t * d),
        "Attention scores": Fraction(layers * heads * t**2),
        "Attention probabilities": Fraction(layers * heads * t**2),
        "Weighted values": Fraction(layers * t * d),
        "Attention output projections": Fraction(layers * t * d),
        "FFN W1/W2, SiLU, product": layers * 4 * t * f,
        "FFN W3 outputs": Fraction(layers * t * d),
        "Final RMSNorm output": Fraction(t * d),
        "LM head logits": Fraction(t * vocab),
        "Cross-entropy log probabilities": Fraction(t * vocab),
    }


def adamw_forward_flops_per_sequence(config: AccountingConfig) -> Fraction:
    """Count dense matrix multiplies, including masked attention entries."""
    t, d, f, layers = config.context_length, config.d_model, config.ff_width, config.num_layers
    return layers * (8 * t * d**2 + 4 * t**2 * d + 6 * t * d * f) + 2 * t * d * config.vocab_size


def adamw_accounting(config: AccountingConfig | None = None) -> None:
    config = AccountingConfig() if config is None else config
    print("AdamW accounting: assignment architecture, FP32, untied embeddings")
    print(
        f"V={config.vocab_size}, T={config.context_length}, L={config.num_layers}, "
        f"d={config.d_model}, h={config.num_heads}, f={config.ff_width}"
    )
    print("Activations: listed operator outputs retained once; analytical estimate, not measured peak.")
    print("Units: GB = 10^9 bytes. No checkpointing, fused attention, or allocator overhead.\n")

    print("(a) Parameter counts")
    counts = adamw_parameter_counts(config)
    for name, count in counts.items():
        print(f"{name:<40} {float(count):>22,.3f}")
    parameters = sum(counts.values())
    print(f"Total parameters: {float(parameters):,.0f}")

    print("\nActivation elements per sequence")
    for name, count in activation_counts_per_sequence(config).items():
        print(f"{name:<40} {float(count):>22,.3f}")
    activation_elements = sum(activation_counts_per_sequence(config).values())
    parameter_bytes = 4 * parameters
    activation_bytes = 4 * activation_elements
    fixed_bytes = 4 * parameter_bytes
    print(f"Parameters: {float(parameter_bytes / 10**9):.9f} GB")
    print(f"Gradients: {float(parameter_bytes / 10**9):.9f} GB")
    print(f"Optimizer m/v: {float(2 * parameter_bytes / 10**9):.9f} GB")
    print(f"Activations per batch item: {float(activation_bytes / 10**9):.9f} GB")

    print("\n(b) Memory budget")
    print(f"M(B) = {float(activation_bytes / 10**9):.9f} * B + {float(fixed_bytes / 10**9):.9f} GB")
    max_batch = max(0, int((80 * 10**9 - fixed_bytes) // activation_bytes))
    for batch in (max_batch, max_batch + 1):
        print(f"B={batch}: {float((fixed_bytes + batch * activation_bytes) / 10**9):.9f} GB")
    print(f"Maximum batch size under this estimate: {max_batch}")

    print("\n(c) AdamW update FLOPs")
    # Literal scalar-operation convention: decay 2, m 3, v 4, update 5 per element.
    optimizer_flops = 14 * parameters
    print("14 * P: weight decay 2, first moment 3, second moment 4, parameter update 5.")
    print("Scalar hyperparameter arithmetic is excluded; sqrt/divide each count as one operation.")
    print(f"Optimizer update: {float(optimizer_flops):,.0f} FLOPs")
    print(f"Precomputing the decay multiplier reduces this to 13 * P = {float(13 * parameters):,.0f} FLOPs.")

    print("\n(d) Training time estimate")
    forward = 1024 * adamw_forward_flops_per_sequence(config)
    training_step = 3 * forward + optimizer_flops
    effective_flops = Fraction(495 * 10**12, 2)
    hours = 400_000 * training_step / effective_flops / 3600
    print(f"Forward per sequence: {float(forward / 1024):,.0f} FLOPs")
    print(f"Forward at B=1024: {float(forward / 10**12):.9f} TFLOPs")
    print(f"Forward + backward + AdamW per step: {float(training_step / 10**12):.9f} TFLOPs")
    print(f"Time per step at 247.5 TFLOP/s: {float(training_step / effective_flops):.9f} seconds")
    print(f"400,000 steps: {float(hours):.6f} hours = {float(hours / 24):.6f} days")
    print("B=1024 exceeds the memory budget above; timing assumes sufficient memory or accumulation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimizer experiments and analytical accounting.")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="learning_rate_tuning",
        choices=("learning_rate_tuning", "adamw_accounting"),
    )
    parser.add_argument("--d-ff", type=int, help="Override the exact 8*d/3 width, e.g. 4288.")
    args = parser.parse_args()
    if args.experiment == "learning_rate_tuning":
        learning_rate_tuning()
    else:
        adamw_accounting(AccountingConfig(d_ff=args.d_ff))

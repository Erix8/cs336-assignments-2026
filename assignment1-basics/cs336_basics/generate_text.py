import argparse
import math
from pathlib import Path

import torch
from torch import Tensor

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    files = parser.add_argument_group("Checkpoint and tokenizer")
    files.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint (.pt).")
    files.add_argument("--vocab-file", type=Path, required=True, help="Hex-encoded tokenizer vocabulary (.json).")
    files.add_argument("--merges-file", type=Path, required=True, help="Ordered tokenizer merges (.json).")
    files.add_argument(
        "--special-tokens",
        nargs="+",
        default=["<|endoftext|>"],
        help="Special tokens used by the training tokenizer.",
    )
    files.add_argument("--eos-token", default="<|endoftext|>", help="Special token that ends generation.")

    sampling = parser.add_argument_group("Generation")
    sampling.add_argument("--prompt", required=True, help="Non-empty text to continue.")
    sampling.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of generated tokens.")
    sampling.add_argument("--temperature", type=float, default=1.0, help="Positive sampling temperature.")
    sampling.add_argument("--top-p", type=float, default=1.0, help="Nucleus probability threshold in (0, 1].")

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--device", default="mps", help="PyTorch device, e.g. cpu, mps, or cuda:0.")
    runtime.add_argument("--seed", type=int, default=42, help="PyTorch random seed.")

    args = parser.parse_args()
    if not args.prompt:
        parser.error("--prompt must not be empty")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if not args.eos_token or args.eos_token not in args.special_tokens:
        parser.error("--eos-token must be a non-empty token listed in --special-tokens")
    if any(not token for token in args.special_tokens):
        parser.error("--special-tokens must not contain empty tokens")
    return args


def sample_next_token(
    logits: Tensor,
    temperature: float,
    top_p: float,
) -> Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    sorted_probs, sorted_ids = torch.sort(softmax(logits), dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    keep_mask = (cumulative_probs - sorted_probs) < top_p

    filtered_probs = sorted_probs.masked_fill(~keep_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    sampled_index = torch.multinomial(filtered_probs, num_samples=1)
    next_token_id = sorted_ids[sampled_index]

    return next_token_id


@torch.inference_mode()
def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    history = list(prompt_ids)
    generated_ids = []
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        context_ids = history[-context_length:]
        input_ids = torch.tensor(
            context_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        next_token = sample_next_token(next_token_logits, temperature, top_p)
        next_token_id = next_token.item()

        history.append(next_token_id)
        generated_ids.append(next_token_id)
        if next_token_id == eos_token_id:
            break

    return generated_ids


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint:
        raise ValueError("Checkpoint has no model_config; add the original training configuration before generation.")
    model_config = checkpoint["model_config"]
    model = TransformerLM(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()
    args.context_length = model_config["context_length"]

    tokenizer = Tokenizer.from_files(
        args.vocab_file,
        args.merges_file,
        args.special_tokens,
    )
    args.eos_token_id = tokenizer.special_token_to_id[args.eos_token]
    prompt_ids = tokenizer.encode(args.prompt)
    with torch.inference_mode():
        generated_ids = generate(
            model,
            prompt_ids,
            args.max_new_tokens,
            args.context_length,
            args.eos_token_id,
            args.temperature,
            args.top_p,
        )
    generated_token_count = len(generated_ids)
    stopped_on_eos = bool(generated_ids and generated_ids[-1] == args.eos_token_id)
    if stopped_on_eos:
        generated_ids = generated_ids[:-1]
    generated_text = tokenizer.decode(generated_ids)
    print(args.prompt + generated_text)
    print(
        f"\nGeneration stats: prompt_tokens={len(prompt_ids)} "
        f"generated_tokens={generated_token_count} "
        f"generated_text_tokens={len(generated_ids)} "
        f"stop_reason={'eos' if stopped_on_eos else 'max_new_tokens'}"
    )


if __name__ == "__main__":
    main()

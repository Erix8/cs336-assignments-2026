import argparse
import logging
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from cs336_basics.data import data_loading
from cs336_basics.model import Transformer
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model on raw uint16 token files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--train-path",
        type=Path,
        required=True,
        help="Training token file (.bin).",
    )
    data.add_argument(
        "--valid-path",
        type=Path,
        required=True,
        help="Validation token file (.bin).",
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Tokenizer vocabulary size.",
    )
    model.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Tokens per training sequence.",
    )
    model.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Hidden dimension.",
    )
    model.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Transformer block count.",
    )
    model.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Attention heads per block.",
    )
    model.add_argument(
        "--d-ff",
        type=int,
        default=1344,
        help="Feed-forward hidden dimension.",
    )
    model.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="RoPE frequency base.",
    )

    training = parser.add_argument_group("Training")
    training.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Sequences per batch.",
    )
    training.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Total updates, including resumed updates.",
    )
    training.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, e.g. cpu, mps, or cuda:0.",
    )
    training.add_argument(
        "--seed",
        type=int,
        default=42,
        help="NumPy and PyTorch random seed.",
    )

    optimizer = parser.add_argument_group("Optimizer")
    optimizer.add_argument(
        "--max-learning-rate",
        type=float,
        default=3e-4,
        help="Peak learning rate.",
    )
    optimizer.add_argument(
        "--min-learning-rate",
        type=float,
        default=3e-5,
        help="Final cosine learning rate.",
    )
    optimizer.add_argument(
        "--warmup-iters",
        type=int,
        default=100,
        help="Warmup updates; cosine ends at max-steps.",
    )
    optimizer.add_argument(
        "--betas",
        type=float,
        nargs=2,
        metavar=("BETA1", "BETA2"),
        default=(0.9, 0.999),
        help="AdamW moment decay coefficients.",
    )
    optimizer.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay.",
    )
    optimizer.add_argument(
        "--max-l2-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold.",
    )

    logging_args = parser.add_argument_group("Logging and checkpoints")
    logging_args.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log training loss every N updates.",
    )
    logging_args.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate every N updates.",
    )
    logging_args.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="Batches averaged per evaluation.",
    )
    logging_args.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save a checkpoint every N updates.",
    )
    logging_args.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Log and checkpoint directory.",
    )
    logging_args.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint to resume from.",
    )

    args = parser.parse_args()
    for name in (
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "batch_size",
        "max_steps",
        "log_interval",
        "eval_interval",
        "eval_batches",
        "save_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("rope_theta", "max_learning_rate", "min_learning_rate", "weight_decay", "max_l2_norm"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.d_model % args.num_heads != 0:
        parser.error("--d-model must be divisible by --num-heads")
    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("The attention head dimension must be even for RoPE")
    if args.rope_theta <= 0 or args.max_l2_norm <= 0:
        parser.error("--rope-theta and --max-l2-norm must be positive")
    if not 0 <= args.min_learning_rate <= args.max_learning_rate:
        parser.error("Learning rates must satisfy 0 <= min-learning-rate <= max-learning-rate")
    if not 0 <= args.warmup_iters < args.max_steps:
        parser.error("--warmup-iters must satisfy 0 <= warmup-iters < max-steps")
    if not all(0 <= beta < 1 for beta in args.betas):
        parser.error("Both --betas values must be in [0, 1)")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    try:
        torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(f"Invalid --device: {exc}")
    for name in ("train_path", "valid_path", "resume"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file: {path}")
    return args


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    args: argparse.Namespace,
) -> float:
    was_training = model.training
    model.eval()
    try:
        losses = []
        for _ in range(args.eval_batches):
            inputs, targets = data_loading(
                dataset=dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
            )
            logits: Tensor = model(inputs)
            loss = cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            losses.append(loss)
        return torch.stack(losses).mean().item()
    finally:
        model.train(was_training)


def setup_logging(out_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train.log", encoding="utf-8"),
        ],
    )


def load_dataset(path: Path, context_length: int) -> npt.NDArray:
    dataset = np.memmap(path, dtype=np.uint16, mode="r")
    if len(dataset) <= context_length:
        raise ValueError(f"{path} needs at least {context_length + 1} tokens; got {len(dataset)}")
    logger.info("Loaded dataset %s: %d tokens", path, len(dataset))
    return dataset


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: npt.NDArray,
    val_dataset: npt.NDArray,
    args: argparse.Namespace,
    start_step: int,
) -> None:
    model.train()
    completed_steps = start_step
    model_config = {
        name: getattr(args, name)
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
    }

    for step in range(start_step, args.max_steps):
        lr = lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()

        inputs, targets = data_loading(
            dataset=train_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )
        logits = model(inputs)
        loss = cross_entropy(
            inputs=logits.reshape(-1, logits.shape[-1]),
            targets=targets.reshape(-1),
        )
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        optimizer.step()
        completed_steps = step + 1

        if completed_steps % args.log_interval == 0:
            logger.info("step=%d/%d lr=%.6g train_loss=%.4f", completed_steps, args.max_steps, lr, loss.item())

        if completed_steps % args.eval_interval == 0:
            val_loss = evaluate(model, val_dataset, args)
            logger.info("step=%d val_loss=%.4f", completed_steps, val_loss)

        if completed_steps % args.save_interval == 0:
            checkpoint_path = args.out_dir / f"step_{completed_steps}.pt"
            save_checkpoint(model, optimizer, completed_steps, checkpoint_path, model_config=model_config)
            logger.info("Saved checkpoint: %s", checkpoint_path)

    checkpoint_path = args.out_dir / "final.pt"
    save_checkpoint(model, optimizer, completed_steps, checkpoint_path, model_config=model_config)
    logger.info("Training finished at step %d; saved %s", completed_steps, checkpoint_path)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out_dir)
    logger.info("Configuration: %s", vars(args))

    train_dataset = load_dataset(args.train_path, args.context_length)
    val_dataset = load_dataset(args.valid_path, args.context_length)

    model = Transformer(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
    )
    model.to(device=args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        logger.info("Resumed from %s at step %d", args.resume, start_step)
    train(model, optimizer, train_dataset, val_dataset, args, start_step)


if __name__ == "__main__":
    main()

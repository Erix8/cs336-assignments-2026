import argparse
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

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
    model.add_argument(
        "--use-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply RoPE to attention queries and keys.",
    )
    model.add_argument(
        "--ffn-type",
        choices=("swiglu", "silu"),
        default="swiglu",
        help="Feed-forward variant; use --d-ff 2048 for SiLU at d-model 512.",
    )
    model.add_argument(
        "--use-rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RMSNorm in every block and before the output projection.",
    )
    model.add_argument(
        "--norm-position",
        choices=("pre", "post"),
        default="pre",
        help="Apply block RMSNorm before each sublayer or after its residual addition.",
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
        "--eval-batch-size",
        type=int,
        default=16,
        help="Sequences per validation batch, independent of training batch size.",
    )
    logging_args.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Seed for fixed validation samples, independent of training sampling.",
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
    logging_args.add_argument("--wandb-project", help="Enable W&B tracking in this project.")
    logging_args.add_argument("--wandb-entity", help="W&B user or team; defaults to the configured entity.")
    logging_args.add_argument("--run-name", help="Display name for this W&B run.")
    logging_args.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
        help="W&B sync mode; only used when --wandb-project is set.",
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
        "eval_batch_size",
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
    if not 0 <= args.eval_seed < 2**32:
        parser.error("--eval-seed must be in [0, 2**32)")
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
    # Restart a private generator so every evaluation sees the same validation windows.
    rng = np.random.default_rng(args.eval_seed)
    model.eval()
    try:
        losses = []
        for _ in range(args.eval_batches):
            inputs, targets = get_batch(
                dataset=dataset,
                batch_size=args.eval_batch_size,
                context_length=args.context_length,
                device=args.device,
                rng=rng,
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


def synchronize_device(device: str) -> None:
    # Accelerator execution is asynchronous, so timing must wait for queued work.
    device_type = torch.device(device).type
    if device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize(device)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: npt.NDArray,
    val_dataset: npt.NDArray,
    args: argparse.Namespace,
    start_step: int,
    run: "Run | None" = None,
) -> None:
    model.train()
    completed_steps = start_step
    model_config = {
        name: getattr(args, name)
        for name in (
            "vocab_size",
            "context_length",
            "d_model",
            "num_layers",
            "num_heads",
            "d_ff",
            "rope_theta",
            "use_rmsnorm",
            "norm_position",
            "use_rope",
            "ffn_type",
        )
    }
    synchronize_device(args.device)
    started_at = interval_started_at = perf_counter()
    last_logged_step = start_step
    tokens_per_step = args.batch_size * args.context_length

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

        inputs, targets = get_batch(
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

        log_training = completed_steps % args.log_interval == 0 or completed_steps == args.max_steps
        metrics = {}
        if log_training:
            synchronize_device(args.device)
            now = perf_counter()
            interval_steps = completed_steps - last_logged_step
            seconds_per_step = (now - interval_started_at) / interval_steps
            train_loss = loss.item()
            metrics.update(
                {
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "perf/seconds_per_step": seconds_per_step,
                    "perf/tokens_per_second": tokens_per_step / seconds_per_step,
                }
            )
            logger.info(
                "step=%d/%d lr=%.6g train_loss=%.4f elapsed_s=%.2f s/step=%.4f tokens/s=%.0f",
                completed_steps,
                args.max_steps,
                lr,
                train_loss,
                now - started_at,
                seconds_per_step,
                tokens_per_step / seconds_per_step,
            )
            interval_started_at = now
            last_logged_step = completed_steps

        if completed_steps % args.eval_interval == 0:
            val_loss = evaluate(model, val_dataset, args)
            logger.info("step=%d val_loss=%.4f", completed_steps, val_loss)
            metrics["val/loss"] = val_loss

        if metrics and run is not None:
            metrics.update(
                {
                    "train/step": completed_steps,
                    "train/tokens_seen": completed_steps * tokens_per_step,
                    "time/elapsed_seconds": perf_counter() - started_at,
                }
            )
            run.log(metrics)

        if completed_steps % args.save_interval == 0:
            checkpoint_path = args.out_dir / f"step_{completed_steps}.pt"
            save_checkpoint(model, optimizer, completed_steps, checkpoint_path, model_config=model_config)
            logger.info("Saved checkpoint: %s", checkpoint_path)

    checkpoint_path = args.out_dir / "final.pt"
    save_checkpoint(model, optimizer, completed_steps, checkpoint_path, model_config=model_config)
    logger.info("Training finished at step %d; saved %s", completed_steps, checkpoint_path)
    if run is not None:
        run.summary["completed_steps"] = completed_steps
        run.summary["session_elapsed_seconds"] = perf_counter() - started_at


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out_dir)
    logger.info("Configuration: %s", vars(args))

    train_dataset = load_dataset(args.train_path, args.context_length)
    val_dataset = load_dataset(args.valid_path, args.context_length)

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        use_rmsnorm=args.use_rmsnorm,
        norm_position=args.norm_position,
        use_rope=args.use_rope,
        ffn_type=args.ffn_type,
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
    tracking = nullcontext(None)
    if args.wandb_project:
        import wandb

        config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
        config["start_step"] = start_step
        config["parameter_count"] = sum(p.numel() for p in model.parameters())
        tracking = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=config,
            dir=str(args.out_dir.resolve()),
            mode=args.wandb_mode,
        )
    with tracking as run:
        if run is not None:
            run.define_metric("train/step")
            run.define_metric("time/elapsed_seconds")
            for prefix in ("train/*", "val/*", "perf/*"):
                run.define_metric(prefix, step_metric="train/step")
            run.define_metric("val/loss", step_metric="train/step", summary="min")
        train(model, optimizer, train_dataset, val_dataset, args, start_step, run=run)


if __name__ == "__main__":
    main()

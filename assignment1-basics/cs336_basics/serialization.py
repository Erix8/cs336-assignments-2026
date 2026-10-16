import os
import typing

import numpy as np
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model_config: dict[str, int | float | bool | str] | None = None,
):
    obj = {}
    obj["model"] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    obj["iteration"] = iteration
    if model_config is not None:
        obj["model_config"] = model_config.copy()
    obj["numpy_rng_state"] = np.random.get_state()
    obj["torch_rng_state"] = torch.get_rng_state()
    if torch.backends.mps.is_available():
        obj["mps_rng_state"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        obj["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src, weights_only=False)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    np.random.set_state(obj["numpy_rng_state"])
    torch.set_rng_state(obj["torch_rng_state"])
    if "mps_rng_state" in obj and torch.backends.mps.is_available():
        torch.mps.set_rng_state(obj["mps_rng_state"])
    if "cuda_rng_states" in obj and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(obj["cuda_rng_states"])
    return obj["iteration"]

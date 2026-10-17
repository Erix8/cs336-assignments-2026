import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_starts = np.random.randint if rng is None else rng.integers
    starts = sample_starts(
        low=0,
        high=len(dataset) - context_length,
        size=batch_size,
    )
    input_slices = [dataset[s : s + context_length] for s in starts]
    target_slices = [dataset[s + 1 : s + context_length + 1] for s in starts]
    inputs_np = np.stack(input_slices)
    targets_np = np.stack(target_slices)
    inputs = torch.tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.tensor(targets_np, dtype=torch.long, device=device)
    return inputs, targets

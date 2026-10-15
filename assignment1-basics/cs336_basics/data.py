import numpy as np
import numpy.typing as npt
import torch


def data_loading(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(
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

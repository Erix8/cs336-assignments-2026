import pytest
import torch

from cs336_basics.nn_utils import gradient_clipping, softmax
from cs336_basics.optimizer import AdamW, lr_cosine_schedule


@pytest.mark.parametrize("kind", ["below-threshold", "zero", "missing", "empty"])
def test_gradient_clipping_leaves_small_or_missing_gradients_unchanged(kind):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    if kind == "below-threshold":
        parameter.grad = torch.tensor([0.03, 0.04])
    elif kind == "zero":
        parameter.grad = torch.zeros_like(parameter)
    before = None if parameter.grad is None else parameter.grad.clone()
    gradient_clipping(iter([] if kind == "empty" else [parameter]), max_l2_norm=1.0)
    if before is None:
        assert parameter.grad is None
    else:
        torch.testing.assert_close(parameter.grad, before, atol=0, rtol=0)


def test_gradient_clipping_uses_one_global_norm():
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
    parameters[0].grad = torch.tensor([3.0, 0.0])
    parameters[1].grad = torch.tensor([0.0, 4.0])
    gradient_clipping(iter(parameters), max_l2_norm=1.0)
    torch.testing.assert_close(parameters[0].grad, torch.tensor([0.6, 0.0]))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([0.0, 0.8]))
    assert parameters[2].grad is None
    norm = torch.linalg.vector_norm(torch.cat([p.grad for p in parameters if p.grad is not None]))
    assert norm <= 1.0


def test_adamw_missing_gradient_skips_parameter_and_existing_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = AdamW([parameter], lr=0.1, weight_decay=0.2)
    optimizer.step()
    assert parameter not in optimizer.state
    torch.testing.assert_close(parameter, torch.tensor([1.0, -2.0]), atol=0, rtol=0)
    parameter.grad = torch.tensor([0.5, -0.5])
    optimizer.step()
    parameter_before = parameter.detach().clone()
    state_before = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in optimizer.state[parameter].items()
    }
    parameter.grad = None
    optimizer.step()
    torch.testing.assert_close(parameter, parameter_before, atol=0, rtol=0)
    for key, value in state_before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(optimizer.state[parameter][key], value, atol=0, rtol=0)
        else:
            assert optimizer.state[parameter][key] == value


def test_adamw_zero_gradient_applies_group_specific_weight_decay():
    parameters = [torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64)) for _ in range(2)]
    optimizer = AdamW(
        [
            {"params": [parameters[0]], "lr": 0.1, "weight_decay": 0.2},
            {"params": [parameters[1]], "lr": 0.03, "weight_decay": 0.5},
        ]
    )
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    for _ in range(3):
        optimizer.step()
    for parameter, factor in zip(parameters, [0.98, 0.985], strict=True):
        torch.testing.assert_close(parameter, torch.tensor([1.0, -2.0], dtype=torch.float64) * factor**3)
        state = optimizer.state[parameter]
        assert state["t"] == 3
        assert torch.count_nonzero(state["m"]) == 0
        assert torch.count_nonzero(state["v"]) == 0


@pytest.mark.parametrize("step, expected", [(0, 1.0), (5, 0.55), (10, 0.1), (11, 0.1)])
def test_cosine_schedule_without_warmup(step, expected):
    actual = lr_cosine_schedule(
        it=step,
        max_learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_iters=0,
        cosine_cycle_iters=10,
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_normalizes_requested_dimension(dim):
    inputs = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) * 100
    actual = softmax(inputs, dim=dim)
    torch.testing.assert_close(actual, torch.softmax(inputs, dim=dim))
    torch.testing.assert_close(actual.sum(dim=dim), torch.ones_like(actual.sum(dim=dim)))

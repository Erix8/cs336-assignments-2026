from argparse import Namespace

import numpy as np
import pytest
import torch

from cs336_basics.data import data_loading
from cs336_basics.train import evaluate, parse_args


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, inputs):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.batches.append(inputs.clone())
        return torch.nn.functional.one_hot(inputs, num_classes=128).float()


def eval_args(**kwargs):
    return Namespace(**dict(eval_batch_size=3, eval_batches=4, eval_seed=42, context_length=5, device="cpu") | kwargs)


def test_evaluation_reuses_samples_across_training_batches_and_random_states():
    dataset = np.arange(128)
    model = RecordingModel()
    args = eval_args(batch_size=1)
    first_loss = evaluate(model, dataset, args)
    first_batches = model.batches.copy()
    np.random.random(100)
    torch.rand(100)
    model.batches.clear()
    args.batch_size = 48
    assert evaluate(model, dataset, args) == first_loss
    assert len(model.batches) == args.eval_batches
    for first, second in zip(first_batches, model.batches, strict=True):
        assert first.shape == (args.eval_batch_size, args.context_length)
        torch.testing.assert_close(first, second)
    model.batches.clear()
    args.eval_seed += 1
    evaluate(model, dataset, args)
    assert any(not torch.equal(a, b) for a, b in zip(first_batches, model.batches, strict=True))


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_evaluation_preserves_rng_and_model_mode(training, fail):
    class FailingModel(RecordingModel):
        def forward(self, inputs):
            raise RuntimeError("Evaluation failed")

    model = FailingModel() if fail else RecordingModel()
    model.train(training)
    dataset = np.arange(128)
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    expected_batch = data_loading(dataset, 7, 5, "cpu")
    np.random.set_state(numpy_state)
    if fail:
        with pytest.raises(RuntimeError, match="Evaluation failed"):
            evaluate(model, dataset, eval_args())
    else:
        evaluate(model, dataset, eval_args())
    actual_state = np.random.get_state()
    assert actual_state[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_state[1], numpy_state[1])
    assert actual_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert model.training is training
    actual_batch = data_loading(dataset, 7, 5, "cpu")
    for expected, actual in zip(expected_batch, actual_batch, strict=True):
        torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("extra", [[], ["--eval-batch-size", "0"], ["--eval-seed", "-1"]])
def test_evaluation_cli(monkeypatch, tmp_path, extra):
    dataset = tmp_path / "tokens.bin"
    np.arange(128, dtype=np.uint16).tofile(dataset)
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--train-path", str(dataset), "--valid-path", str(dataset), "--vocab-size", "128", *extra],
    )
    if extra:
        with pytest.raises(SystemExit) as exc:
            parse_args()
        assert exc.value.code == 2
    else:
        args = parse_args()
        assert args.eval_batch_size == 16
        assert args.eval_seed == 42

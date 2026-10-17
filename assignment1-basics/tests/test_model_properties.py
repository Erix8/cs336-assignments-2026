import pytest
import torch
import torch.nn.functional as F

from cs336_basics.model import FeedForward, RMSNorm, TransformerLM, scaled_dot_product_attention
from cs336_basics.nn_utils import cross_entropy


@pytest.fixture(autouse=True)
def fixed_torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        yield


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "variant",
    [{}, {"use_rope": False}, {"use_rmsnorm": False}, {"norm_position": "post"}, {"ffn_type": "silu"}],
    ids=["baseline", "nope", "no-norm", "post-norm", "silu"],
)
def test_transformer_future_tokens_do_not_change_prefix(batch_size, variant):
    model = TransformerLM(
        vocab_size=32,
        context_length=6,
        d_model=16,
        num_layers=2,
        num_heads=2,
        d_ff=32,
        rope_theta=10000,
        **variant,
    ).eval()
    tokens = torch.randint(32, (batch_size, 6))
    changed = tokens.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % 32
    with torch.no_grad():
        original = model(tokens)
        perturbed = model(changed)
        prefix_only = model(tokens[:, :3])
    torch.testing.assert_close(original[:, :3], perturbed[:, :3], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(original[:, :3], prefix_only, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(original[:, 3:], perturbed[:, 3:])


@pytest.mark.parametrize("scale", [1, 1000])
def test_cross_entropy_gradients_match_pytorch(scale):
    logits = (torch.randn(7, 11, dtype=torch.float64) * scale).requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    targets = torch.randint(11, (7,))
    actual = cross_entropy(logits, targets)
    expected = F.cross_entropy(reference_logits, targets)
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual, logits)
    (expected_grad,) = torch.autograd.grad(expected, reference_logits)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize("masked", [False, True])
def test_attention_outputs_and_gradients_match_pytorch(masked):
    tensors = [torch.randn(2, 2, 4, width, dtype=torch.float64, requires_grad=True) for width in (3, 3, 5)]
    references = [x.detach().clone().requires_grad_() for x in tensors]
    mask = torch.ones(4, 4, dtype=torch.bool).tril() if masked else None
    actual = scaled_dot_product_attention(*tensors, mask=mask)
    expected = F.scaled_dot_product_attention(*references, attn_mask=mask, dropout_p=0.0)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-9)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, tensors, grad_outputs=upstream)
    expected_grads = torch.autograd.grad(expected, references, grad_outputs=upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize("ffn_type", ["swiglu", "silu"])
def test_feedforward_input_and_weight_gradients_match_pytorch(ffn_type):
    model = FeedForward(8, d_ff=12, ffn_type=ffn_type, dtype=torch.float64)
    inputs = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    parameters = dict(model.named_parameters())
    weights = {name: p.detach().clone().requires_grad_() for name, p in parameters.items()}
    hidden = F.silu(F.linear(reference_inputs, weights["w1.weight"]))
    if ffn_type == "swiglu":
        hidden = hidden * F.linear(reference_inputs, weights["w3.weight"])
    expected = F.linear(hidden, weights["w2.weight"])
    actual = model(inputs)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, [inputs, *parameters.values()], grad_outputs=upstream)
    expected_grads = torch.autograd.grad(expected, [reference_inputs, *weights.values()], grad_outputs=upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize("scale", [0.0, 1e-6, 1.0])
def test_rmsnorm_outputs_and_gradients_match_pytorch(scale):
    model = RMSNorm(8)
    with torch.no_grad():
        model.weight.copy_(torch.randn(8))
    # RMSNorm deliberately computes in float32, so the reference uses that precision too.
    inputs = (torch.randn(2, 3, 8) * scale).requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = model.weight.detach().clone().requires_grad_()
    actual = model(inputs)
    expected = F.rms_norm(reference_inputs, (8,), reference_weight, eps=model.eps)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, [inputs, model.weight], grad_outputs=upstream)
    expected_grads = torch.autograd.grad(expected, [reference_inputs, reference_weight], grad_outputs=upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)

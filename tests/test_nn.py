import numpy as np

from src.nn import Dropout
from src.tensor import Tensor
from src.transformer.layer.norm.norm import LayerNorm


def test_dropout_backward():
    x = Tensor(np.ones((4, 4)), requires_grad=True)
    drop = Dropout(p=0.25)
    y = drop(x)
    y.backward(np.ones_like(y.data))

    # values should be either 0 or 1 / (1 - p)
    scale = 1.0 / (1.0 - 0.25)
    expected = ((y.data != 0) * scale).astype(x.data.dtype)
    assert np.allclose(x.grad, expected), "Dropout backward must scale mask correctly"


def test_dropout_training_mode():
    x = Tensor(np.ones((1000,)), requires_grad=True)
    drop = Dropout(p=0.5)
    y = drop(x)
    # expect ~50% to be 0 and ~50% to be 2.0 (1 / (1 - 0.5) = 2)
    num_zeros = np.sum(y.data == 0.0)
    num_twos = np.sum(y.data == 2.0)
    assert abs(num_zeros - 500) < 100, "Dropout zeros roughly 50%"
    assert abs(num_twos - 500) < 100, "Remaining values scaled properly"


def test_layernorm_forward_shape():
    x = Tensor(np.random.randn(4, 10), requires_grad=True)
    norm = LayerNorm(d_model=10, eps=1e-5)
    out = norm(x)
    assert out.data.shape == x.data.shape, "Output shape must match input shape"


def test_layernorm_forward_mean_std():
    x = Tensor(np.random.randn(2, 4), requires_grad=True)
    norm = LayerNorm(d_model=4, eps=1e-5)
    out = norm(x)

    # remove scale/shift to test normalised values
    unscaled = (out - norm.beta) / norm.gamma

    # mean and variance across last dim
    mean = unscaled.data.mean(axis=-1)
    std = unscaled.data.std(axis=-1)

    assert np.allclose(mean, 0, atol=1e-3), f"Mean not zero: {mean}"
    assert np.allclose(std, 1, atol=1e-3), f"Std not one: {std}"


def test_layernorm_backward_passes():
    x = Tensor(np.random.randn(3, 5), requires_grad=True)
    norm = LayerNorm(d_model=5, eps=1e-5)
    out = norm(x)
    out.sum().backward()

    assert x.grad is not None, "Gradients must be propagated to input"
    assert norm.gamma.grad is not None, "Gradients must flow to gamma"
    assert norm.beta.grad is not None, "Gradients must flow to beta"


if __name__ == '__main__':
    test_dropout_training_mode()
    test_dropout_backward()
    test_layernorm_forward_shape()
    test_layernorm_forward_mean_std()
    test_layernorm_backward_passes()

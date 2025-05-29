from src.tensor import Tensor

import numpy as np

from src.transformer.layer.feed_forward.position_wise_feed_forward import PositionWiseFeedForwardLayer


def test_positionwise_ff_output_shape_and_type():
    x = Tensor(np.random.randn(2, 5, 16), requires_grad=True)  # (batch, seq_len, d_model)
    ff = PositionWiseFeedForwardLayer(d_model=16, dff=32)
    out = ff(x)
    assert isinstance(out, Tensor)
    assert out.shape() == x.shape()


def test_positionwise_ff_backward_pass():
    x = Tensor(np.random.randn(1, 3, 8), requires_grad=True)
    ff = PositionWiseFeedForwardLayer(d_model=8, dff=16)
    out = ff(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape()


def test_positionwise_ff_numerical_gradient():
    x = Tensor(np.random.randn(1, 2, 4), requires_grad=True)
    ff = PositionWiseFeedForwardLayer(d_model=4, dff=8, p_drop=0.0)

    out = ff(x)
    out.sum().backward()
    analytical_grad = np.array(x.grad.data)

    epsilon = 1e-4
    numerical_grad = np.zeros_like(x.data)

    for i in range(x.data.shape[1]):
        for j in range(x.data.shape[2]):
            x_pos = Tensor(x.data.copy(), requires_grad=True)
            x_pos.data[0, i, j] += epsilon
            y_pos = ff(x_pos).sum().data

            x_neg = Tensor(x.data.copy(), requires_grad=True)
            x_neg.data[0, i, j] -= epsilon
            y_neg = ff(x_neg).sum().data

            numerical_grad[0, i, j] = (y_pos - y_neg) / (2 * epsilon)

    diff = np.abs(numerical_grad - analytical_grad)
    max_diff = diff.max()
    assert max_diff < 1e-2, f"Gradient check failed: {max_diff}"


if __name__ == '__main__':
    test_positionwise_ff_output_shape_and_type()
    test_positionwise_ff_backward_pass()
    test_positionwise_ff_numerical_gradient()

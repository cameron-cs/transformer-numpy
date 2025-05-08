import numpy as np

from src.tensor import Tensor


def test_tensor_transpose():
    x = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
    y = x.transpose(0, 2, 1, 3)  # shape: (2, 4, 3, 5)
    z = y.sum()
    z.backward()
    assert x.grad.shape == x.shape()  # should match original shape (2, 3, 4, 5)


if __name__ == '__main__':
    test_tensor_transpose()

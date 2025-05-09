import numpy as np

from src.tensor import Tensor


def test_tensor_transpose():
    x = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
    y = x.transpose(0, 2)  # (4, 3, 2, 5)
    z = y.sum()
    z.backward()
    assert x.grad.shape == x.shape()  # should match original shape (4, 3, 2, 5)


def test_masked_fill():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    mask = Tensor([[1, 0], [0, 1]]).float()
    filled = a.masked_fill(mask, float('-inf'))

    expected = np.where(mask.data.astype(bool), a.data, float('-inf'))
    assert np.allclose(filled.data, expected), f"Expected {expected}, got {filled.data}"

    filled.backward(np.ones_like(filled.data))

    expected_grad = np.where(mask.data.astype(bool), 1.0, 0.0)
    assert np.allclose(a.grad, expected_grad), f"Expected grad {expected_grad}, got {a.grad}"


def test_divide():
    a = Tensor([[4.0, 9.0], [16.0, 25.0]], requires_grad=True)
    b = Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    c = a / b

    # forward check
    expected = np.array([[2.0, 3.0], [4.0, 5.0]])
    assert np.allclose(c.data, expected), f"Expected {expected}, got {c.data}"

    # backward check
    c.backward(np.ones_like(c.data))

    expected_grad_a = 1 / b.data
    expected_grad_b = -a.data / (b.data ** 2)
    assert np.allclose(a.grad, expected_grad_a), f"Expected grad {expected_grad_a}, got {a.grad}"
    assert np.allclose(b.grad, expected_grad_b), f"Expected grad {expected_grad_b}, got {b.grad}"


if __name__ == '__main__':
    test_tensor_transpose()
    test_masked_fill()
    test_divide()

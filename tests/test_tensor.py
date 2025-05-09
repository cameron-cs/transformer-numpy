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


def test_view_basic():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = a.view(4)

    assert isinstance(b, Tensor)
    assert b.shape() == (4,)
    np.testing.assert_array_equal(b.data, np.array([1.0, 2.0, 3.0, 4.0]))


def test_view_backward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = a.view(4)
    c = b * 2.0
    d = c.sum()
    d.backward()

    # gradient should be 2.0 for all elements in the original shape
    expected_grad = np.array([[2.0, 2.0], [2.0, 2.0]])
    np.testing.assert_array_equal(a.grad, expected_grad)


def test_view_gradient_accumulates():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = a.view(4)
    c = b * b  # square all elements
    d = c.sum()
    d.backward()

    # each grad is 2*x for x in a
    expected_grad = 2 * a.data
    np.testing.assert_array_equal(a.grad, expected_grad)


def test_view_chain_with_reshape():
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    b = a.view(-1)  # Flatten
    c = b.reshape(2, 3, 4)
    d = c.sum()
    d.backward()

    np.testing.assert_array_equal(a.grad, np.ones_like(a.data))


def test_contiguous_preserves_data_and_grad():
    a = Tensor(np.random.randn(4, 5), requires_grad=True)
    b = a.T.contiguous()

    assert b.data.flags['C_CONTIGUOUS'], "b must be contiguous"
    assert np.allclose(a.data.T, b.data), "Data mismatch after .contiguous()"

    # test backward
    loss = b.sum()
    loss.backward()
    assert a.grad is not None, "Grad should not be None"
    assert np.allclose(a.grad, np.ones_like(a.data)), "Incorrect gradient"


def test_contiguous_noop_on_already_contiguous():
    a = Tensor(np.random.randn(3, 3), requires_grad=True)
    b = a.contiguous()
    assert a is b, "Should return self if already contiguous"


def test_mean_axis_none():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    y = x.mean()
    y.backward()
    assert np.allclose(x.grad, np.ones_like(x.data) / 6)


def test_mean_axis_0_keepdim():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    y = x.mean(axis=0, keepdim=True)
    y.backward(np.ones_like(y.data))
    assert x.grad.shape == x.data.shape
    assert np.allclose(x.grad, np.ones_like(x.data) / 2)


def test_mean_axis_1_no_keepdim():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    y = x.mean(axis=1)
    y.backward(np.ones_like(y.data))
    assert x.grad.shape == x.data.shape
    assert np.allclose(x.grad, np.ones_like(x.data) / 3)


def test_var_axis_1_unbiased_false_keepdim_true():
    x = Tensor(np.array([[1.0, 2.0, 3.0],
                         [2.0, 2.0, 2.0]]), requires_grad=True)
    y = x.var(axis=1, keepdim=True, unbiased=False)
    y.backward(np.ones_like(y.data))

    # manually compute expected gradients
    # y = ((x - mean) ** 2).mean()
    # dy/dx = 2 * (x - mean) / N
    expected_grad = (x.data - x.data.mean(axis=1, keepdims=True)) * (2 / 3)
    assert np.allclose(x.grad, expected_grad)


def test_sqrt_forward_backward():
    x_data = np.array([[4.0, 9.0], [16.0, 25.0]])
    x = Tensor(x_data, requires_grad=True)

    y = x.sqrt()
    y.backward(np.ones_like(x_data))

    expected_forward = np.sqrt(x_data)
    expected_grad = 0.5 / np.sqrt(x_data)

    assert np.allclose(y.data, expected_forward)
    assert np.allclose(x.grad, expected_grad)


if __name__ == '__main__':
    test_tensor_transpose()
    test_masked_fill()
    test_divide()
    test_view_basic()
    test_view_backward()
    test_view_gradient_accumulates()
    test_view_chain_with_reshape()
    test_contiguous_noop_on_already_contiguous()
    test_contiguous_preserves_data_and_grad()
    test_mean_axis_none()
    test_mean_axis_0_keepdim()
    test_mean_axis_1_no_keepdim()
    test_var_axis_1_unbiased_false_keepdim_true()
    test_sqrt_forward_backward()

from typing import Union, Tuple, List, Optional
import numpy as np

from src.nn_utils import unbroadcast


class Tensor:
    """
    A class that represents a Tensor with automatic differentiation features.
    The class supports basic tensor operations like addition, subtraction, multiplication, etc.

    Attributes:
        data (np.ndarray): The data stored in the tensor.
        requires_grad (bool): Whether or not the tensor requires gradient computation.
        grad (np.ndarray or None): The gradient of the tensor (None if no gradients are required).
        _backward (callable): The function that defines the backward pass for this tensor.
        _prev (set): The set of tensors that contributed to this tensor.
        _op (str): The operation that created this tensor.
    """

    def __init__(self, data: Union[np.ndarray, float, int], requires_grad: bool = False,
                 _children: Tuple['Tensor', ...] = (), _op: str = '', name: Optional[str]=''):
        """
        Initialise a Tensor object.

        Args:
            data (Union[np.ndarray, float, int]): The data for the tensor.
            requires_grad (bool): Whether or not to track gradients for this tensor.
            _children (Tuple['Tensor', ...]): Tensors that created this one (for backpropagation).
            _op (str): The operation that created this tensor.
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.name = f"Tensor{name if name else id(data)}"

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def requires_grad(self, req_grad):
        self.requires_grad = req_grad

    def prev(self):
        return self._prev

    def op(self):
        return self._op

    def backward(self, grad: np.ndarray = None) -> None:
        """
        Perform backpropagation to compute gradients.

        Args:
            grad (np.ndarray, optional): The gradient of the loss with respect to this tensor.
                                         If None, assumes this tensor is a scalar.
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            grad = np.ones_like(self.data)

        self.grad = self.grad + grad if self.grad is not None else grad

        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Add another tensor or a scalar to this tensor.

        Args:
            other (Union['Tensor', float, int]): The tensor or scalar to add.

        Returns:
            Tensor: A new tensor representing the result of the addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Subtract another tensor or a scalar from this tensor.

        Args:
            other (Union['Tensor', float, int]): The tensor or scalar to subtract.

        Returns:
            Tensor: A new tensor representing the result of the subtraction.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='-')

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += -unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def exp(self) -> 'Tensor':
        """
        Compute the element-wise exponential of this tensor.

        Returns:
            Tensor: A new tensor representing the element-wise exponential of this tensor.
        """
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad,
                     _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                grad = unbroadcast(np.exp(self.data) * out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def log(self) -> 'Tensor':
        """
        Compute the element-wise natural logarithm of this tensor.

        Returns:
            Tensor: A new tensor representing the element-wise logarithm of this tensor.
        """
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad,
                     _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                grad = unbroadcast((1 / self.data) * out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * Tensor(-1.0)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data == other.data)

    def equal(self, other) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data == other.data)

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Multiply this tensor by another tensor or scalar.

        Args:
            other (Union['Tensor', float, int]): The tensor or scalar to multiply.

        Returns:
            Tensor: A new tensor representing the result of the multiplication.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                grad_self = unbroadcast(other.data * out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = unbroadcast(self.data * out.grad, other.data.shape)
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='/')

        def _backward():
            if self.requires_grad:
                grad_self = unbroadcast((1 / other.data) * out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = unbroadcast((-self.data / (other.data ** 2)) * out.grad, other.data.shape)
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def __pow__(self, power: Union[int, float]) -> 'Tensor':
        """
        Raise the tensor to a power element-wise.

        Args:
            power (Union[int, float]): The exponent to raise the tensor to.

        Returns:
            Tensor: A new tensor representing the result of raising this tensor to the given power.
        """
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, _children=(self,), _op=f'**{power}')

        def _backward():
            if self.requires_grad:
                grad_self = unbroadcast((power * self.data ** (power - 1)) * out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = grad_self

        out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication method.

        Args:
            other (Tensor): The tensor to multiply with.

        Returns:
            Tensor: A new tensor representing the result of the matrix multiplication.
        """
        assert isinstance(other, Tensor)
        out = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                grad_self = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad,
                     _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def unsqueeze(self, dim: int) -> 'Tensor':
        data = np.expand_dims(self.data, axis=dim)
        out = Tensor(data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.squeeze(out.grad, axis=dim)

        out._backward = _backward
        out._prev = {self}
        return out

    def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
        if dim is not None:
            assert self.data.shape[dim] == 1, f"Cannot squeeze dim {dim} with size {self.data.shape[dim]}"
            data = np.squeeze(self.data, axis=dim)
        else:
            data = np.squeeze(self.data)

        out = Tensor(data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if dim is not None:
                    grad = np.expand_dims(grad, axis=dim)
                else:
                    # match original shape by broadcasting
                    grad = grad.reshape(self.data.shape)
                self.grad += grad

        out._backward = _backward
        out._prev = {self}
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad, _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                grad_self = np.ones_like(self.data) * out.grad
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdim=False):
        data = self.data.mean(axis=axis, keepdims=keepdim)
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op='mean')

        def _backward():
            if not self.requires_grad:
                return

            # get the shape of the gradient to broadcast properly
            grad = out.grad
            if axis is None:
                div = self.data.size
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple([a if a >= 0 else a + self.data.ndim for a in axes])
                div = np.prod([self.data.shape[a] for a in axes])

                # broadcast grad back to input shape
                if not keepdim:
                    grad = np.expand_dims(grad, axis=axes)

            grad_self = grad * (1.0 / div) * np.ones_like(self.data)

            if self.grad is None:
                self.grad = grad_self
            else:
                self.grad += grad_self

        out._backward = _backward
        return out

    def float(self) -> 'Tensor':
        return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)

    @staticmethod
    def concat(tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        data = np.concatenate([t.data for t in tensors], axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data, requires_grad)

        def _backward():
            if not out.requires_grad:
                return

            sizes = [t.data.shape[axis] for t in tensors]
            splits = np.split(out.grad, np.cumsum(sizes[:-1]), axis=axis)

            for i, t in enumerate(tensors):
                if t.requires_grad:
                    t.grad += splits[i]

        out._backward = _backward
        out._prev = set(t for t in tensors if t.requires_grad)
        return out

    def sqrt(self):
        data = np.sqrt(self.data)
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op='sqrt')

        def _backward():
            if not self.requires_grad:
                return

            # derivative: 1 / (2 * sqrt(x))
            grad_self = (0.5 / np.sqrt(self.data)) * out.grad

            if self.grad is None:
                self.grad = grad_self
            else:
                self.grad += grad_self

        out._backward = _backward
        return out

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        axes = list(range(self.data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        out_data = self.data.transpose(axes)

        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op='transpose')

        def _backward():
            if self.requires_grad:
                inverse = np.argsort(axes)
                grad_self = out.grad.transpose(inverse)
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

        out._backward = _backward
        return out

    @property
    def T(self) -> 'Tensor':
        if len(self.shape()) != 2:
            raise ValueError(f".T only supports 2D tensors, got shape {self.data.shape}")
        return self.transpose(1, 0)

    def shape(self):
        return self.data.shape

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.data.shape
        else:
            return self.data.shape[dim]

    def view(self, *shape: int) -> 'Tensor':
        shape = tuple(shape)
        reshaped = self.data.reshape(shape)
        out = Tensor(reshaped, requires_grad=self.requires_grad, _children=(self,), _op='view')

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def contiguous(self) -> 'Tensor':
        """
        Returns a contiguous copy of the tensor if it is not already contiguous.

        Returns:
            Tensor: A contiguous version of the tensor.
        """
        if self.data.flags['C_CONTIGUOUS']:
            return self  # llready contiguous
        out = Tensor(np.ascontiguousarray(self.data), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                # just propagate the gradient back as is
                self.grad += out.grad

        out._backward = _backward
        out._prev = {self}
        out._op = 'contiguous'
        return out

    def masked_fill(self, mask: 'Tensor', value: Union[float, int]) -> 'Tensor':
        assert self.data.shape == mask.data.shape, "Shapes must match for masked_fill"
        out_data = np.where(mask.data, self.data, value)
        out = Tensor(out_data, requires_grad=self.requires_grad or mask.requires_grad,
                     _children=(self,), _op='masked_fill')

        def _backward():
            if self.requires_grad:
                grad = np.where(mask.data, out.grad, 0)
                self.grad += grad

        out._backward = _backward
        return out

    def var(self, axis=None, keepdim=False, unbiased=True):
        mean = self.mean(axis=axis, keepdim=True)
        diff = self - mean
        squared = diff * diff
        if unbiased:
            axes = axis if isinstance(axis, tuple) else (axis,) if axis is not None else tuple(range(self.data.ndim))
            axes = tuple([a if a >= 0 else a + self.data.ndim for a in axes])
            div = np.prod([self.data.shape[a] for a in axes])
            return squared.sum(axis=axis, keepdim=keepdim) * (1.0 / (div - 1))
        else:
            return squared.mean(axis=axis, keepdim=keepdim)

    @staticmethod
    def zeros(shape) -> 'Tensor':
        return Tensor(np.zeros(shape), requires_grad=True)

    @staticmethod
    def stack(tensors, axis=1) -> 'Tensor':
        data = np.stack([t.data for t in tensors], axis=axis)
        return Tensor(data, requires_grad=True)

    @staticmethod
    def arange(start, stop=None, step=1, dtype=None, requires_grad=False):
        data = np.arange(start, stop, step, dtype=dtype)
        return Tensor(data, requires_grad=requires_grad)

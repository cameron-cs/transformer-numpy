import numpy as np

from src.nn_utils import xavier_init
from src.tensor import Tensor


class Module:
    """
    The base class for all neural network modules (layers).
    Provides a forward method and parameter management.
    """

    def __init__(self):
        self.requires_grad = True
        self._parameters = []

    def forward(self, *input: Tensor) -> Tensor:
        """
        The forward pass for this module.
        Must be overridden by child classes.
        """
        raise NotImplementedError

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        """
        Zero the gradients of all parameters.
        """
        for param in self.parameters():
            param.zero_grad()

    def parameters(self):
        """
        Return a list of parameters (weights and biases) of the module.
        """
        return self._parameters

    def _add_parameter(self, param: Tensor):
        """
        Add a tensor to the list of parameters.
        This is a helper method used by subclasses.
        """
        self._parameters.append(param)


class Linear(Module):
    """
    Fully connected layer (Linear layer).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weights = Tensor(xavier_init((in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

        self._add_parameter(self.weights)
        self._add_parameter(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the linear layer.
        """
        return input.matmul(self.weights) + self.bias


class Sequential(Module):
    """
    A container for layers arranged sequentially.
    """

    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

    def forward(self, input: Tensor) -> Tensor:
        """
        Pass input through the sequential stack of layers.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def parameters(self):
        """
        Returns all the parameters from the contained layers.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class Dropout(Module):
    """
    Dropout layer for regularisation.

    Parameters:
        p (float): Probability of dropping a unit (0 <= p < 1).

    Behavior:
        - During training: zeroes out inputs with probability p, scales by 1/(1-p).
        - During evaluation: returns input unchanged.
    """
    def __init__(self, p: float = 0.5):
        super(Dropout, self).__init__()
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
        self.p = p
        self.training = True  # default mode is training

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        # create dropout mask: keep probability = (1 - p)
        mask = (np.random.rand(*x.data.shape) > self.p).astype(x.data.dtype)
        scale = 1.0 / (1.0 - self.p)
        out_data = x.data * mask * scale
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                grad = out.grad * mask * scale
                x.grad = x.grad + grad if x.grad is not None else grad

        out._backward = _backward
        out._prev = {x}
        out._op = 'dropout'
        return out


class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        weight = np.random.randn(num_embeddings, embedding_dim) * (1.0 / np.sqrt(num_embeddings))
        self.weight = Tensor(weight, requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        indices = input.data.astype(int)  # or np.int64
        # indices: shape (N,) or (B, T) → use advanced indexing
        out_data = self.weight.data[indices]
        out = Tensor(out_data, requires_grad=self.weight.requires_grad)

        # save backward info
        def _backward():
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                    # scatter-add gradients for repeated indices
                np.add.at(self.weight.grad, indices, out.grad)

        out._backward = _backward
        out._prev = [self.weight]
        out._op = 'embedding'
        return out


class Softmax(Module):
    """
     Applies the softmax function to the input tensor along the specified axis.

     Softmax converts raw logits into probabilities by exponentiating and normalising them.
     The output probabilities sum to 1 along the given axis.

     Parameters:
         axis (int): Axis over which to compute softmax. Default is -1 (last dimension).

     Forward:
         Input: Tensor of shape (batch_size, num_classes) or similar.
         Output: Same shape, with probabilities in range (0, 1), summing to 1 across `axis`.

     Backward:
         Computes the Jacobian-vector product for each sample in the batch using the softmax Jacobian:
             J = diag(p) - p @ p.T
     """

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        # shift logits for numerical stability
        shifted = x.data - np.max(x.data, axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted)
        probs = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        out = Tensor(probs, requires_grad=x.requires_grad)

        def _backward():
            if not x.requires_grad:
                return

            grad = out.grad
            dx = np.zeros_like(x.data)

            # reshape to (batch*, softmax_dim)
            flat_probs = probs.reshape(-1, probs.shape[self.axis])
            flat_grad = grad.reshape(-1, grad.shape[self.axis])

            for i in range(flat_probs.shape[0]):
                p = flat_probs[i].reshape(-1, 1)
                J = np.diagflat(p) - p @ p.T
                dx.reshape(-1, dx.shape[self.axis])[i] = J @ flat_grad[i]

            x.grad = x.grad + dx if x.grad is not None else dx

        out._backward = _backward
        out._prev = {x}
        return out


class ReLU(Module):
    """
    ReLU (Rectified Linear Unit) activation function.
    The ReLU function applies the operation f(x) = max(0, x) element-wise to the input tensor.
    It introduces non-linearity to the model and is commonly used in deep neural networks.
    """

    def forward(self, x: Tensor) -> Tensor:
        out_data = np.maximum(0, x.data)
        out = Tensor(out_data, requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                grad_input = (x.data > 0).astype(float) * out.grad
                x.grad = x.grad + grad_input if x.grad is not None else grad_input

        out._backward = _backward
        out._prev = {x}
        return out

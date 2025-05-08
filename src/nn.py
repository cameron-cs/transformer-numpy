import numpy as np

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

    def __call__(self, *args, **kwargs):
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
            if x.requires_grad:
                # jacobian-based gradient
                dx = np.empty_like(x.data)
                for i in range(x.data.shape[0]):
                    p = probs[i].reshape(-1, 1)
                    J = np.diagflat(p) - np.dot(p, p.T)
                    dx[i] = J @ out.grad[i]
                x.grad += dx

        out._backward = _backward
        out._prev = {x}
        return out

import numpy as np

from src.tensor import Tensor
from src.parameter import Parameter
from src.nn import Module


class LayerNorm(Module):
    """
      Applies normalisation over the last dimension of the input tensor
      (typically the embedding or feature dimension), followed by learnable
      affine transformation using `gamma` and `beta`.

      This layer normalises each sample independently rather than across the batch,
      making it suitable for transformers and non-CNN models.

      Args:
          d_model (int): The size of the last dimension to normalise over.
          eps (float): A small constant added to variance for numerical stability.

      Learnable Parameters:
          - gamma: Scale parameter (initialised to ones)
          - beta: Shift parameter (initialised to zeros)

      Forward:
          Input: Tensor of shape (batch_size, ..., d_model)
          Output: Tensor of the same shape
      """
    def __init__(self, d_model, eps):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(np.ones(d_model))
        self.beta = Parameter(np.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # 1: compute the mean across the last dimension (feature dim)
        # shape: same as x, but with last dim reduced (or kept if keepdim=True)
        mean: Tensor = x.mean(axis=-1, keepdim=True)

        # 2: compute the variance across the last dimension
        # `unbiased=False` uses population variance (divides by N), consistent with PyTorch default
        var: Tensor = x.var(axis=-1, unbiased=False, keepdim=True)

        std: Tensor = (var + self.eps).sqrt()

        # 3: normalise the input
        # (x - mean) / sqrt(var + eps)  -- per sample normalisation
        normed: Tensor = (x - mean) / std

        # 4: apply affine transformation using learnable scale and shift
        # gamma and beta are broadcasted over batch and spatial dims
        return self.gamma * normed + self.beta

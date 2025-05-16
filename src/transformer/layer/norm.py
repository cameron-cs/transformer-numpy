import numpy as np

from src.tensor import Tensor
from src.parameter import Parameter
from src.nn import Module


class NormLayer(Module):
    """
      Applies normalisation over the last dimension of the input tensor
      (typically the embedding or feature dimension), followed by learnable
      affine transformation using `gamma` and `beta`.

      Example (after InputEmbeddingLayer + PositionalEncodingLayer):

        Sentence:        "I love jazz because it is smoother"
        Token Embeddings (dim=4):  [shown below before normalisation]

        Raw Input:
        ┌────────────┬────────────┬────────────┬────────────┐
        │   0.10     │   0.30     │  -0.20     │   0.50     │ ← "I"
        │   0.25     │  -0.10     │   0.40     │   0.05     │ ← "love"
        │  -0.30     │   0.60     │   0.20     │   0.10     │ ← "jazz"
        └────────────┴────────────┴────────────┴────────────┘

        Step-by-step per token (last dim is normalised):

        Token: "I"
        μ = mean([0.1, 0.3, -0.2, 0.5])   = 0.175
        σ² = var([...])                   ≈ 0.0675
        std = sqrt(σ² + eps)              ≈ 0.260

        Normalised:
        [(0.1 - 0.175) / 0.260, (0.3 - 0.175) / 0.260, ...]
        ≈ [-0.29, 0.48, -1.44, 1.25]

        Output after γ and β (default: γ=1, β=0):
        ≈ [-0.29, 0.48, -1.44, 1.25]

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
    def __init__(self, d_model, eps: int = 10**-6):
        super(NormLayer, self).__init__()
        self.gamma = Parameter(np.ones(d_model))
        self.beta = Parameter(np.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # 1: compute the mean across the last dimension (feature dim)
        # shape: same as x, but with last dim reduced (or kept if keepdim=True)
        mean: Tensor = x.mean(axis=-1, keepdim=True)

        # 2: compute the variance across the last dimension
        # `unbiased=False` uses population variance (divides by N)
        var: Tensor = x.var(axis=-1, unbiased=False, keepdim=True)

        std: Tensor = (var + self.eps).sqrt()

        # 3: normalise the input
        # (x - mean) / sqrt(var + eps)  -- per sample normalisation
        normed: Tensor = (x - mean) / std

        # 4: apply affine transformation using learnable scale and shift
        # gamma and beta are broadcasted over batch and spatial dims
        return self.gamma * normed + self.beta

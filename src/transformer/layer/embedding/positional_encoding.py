import numpy as np

from src.nn import Module
from src.tensor import Tensor


class PositionalEncodingLayer(Module):
    """
        Compute the positional encoding for a given position and dimension index
        using the sinusoidal encoding scheme from the Transformer architecture.

        The formulas used are:
            PE(pos, 2i)   = sin(pos / (10000 ** (2i / d_model)))
            PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model)))

        These encode absolute position information into each dimension of
        the input embeddings using sinusoids of varying wavelengths.

        Parameters
        ----------
        d_model : int
            The total dimensionality of the model's embeddings (e.g., 512).

        max_seq_len : int
            The index of the embedding dimension. Must satisfy 0 <= i < d_model.
        """
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncodingLayer, self).__init__()
        pos: Tensor = Tensor.arange(0, max_seq_len).float().unsqueeze(1)  # [max_len, 1]
        two_i: Tensor = Tensor.arange(0, d_model, 2).float()  # [d_model // 2]

        # the divisor term: 10000 ** (2i / d_model)
        denominator = (10000 ** (two_i.data / d_model)).reshape(1, -1)  # [1, d_model // 2]

        # angle rates
        angle_rads = pos.data / denominator  # [max_len, d_model // 2]

        # full encoding matrix
        encoding = np.zeros((max_seq_len, d_model), dtype=np.float32)
        encoding[:, 0::2] = np.sin(angle_rads)  # odd
        encoding[:, 1::2] = np.cos(angle_rads)  # even

        # constant Tensor (no gradients)
        self.encoding = Tensor(encoding, requires_grad=False)

    def forward(self, x: Tensor):
        """
         Returns the positional encoding slice to be added to input embeddings.

         :param x: Tensor of shape [batch_size, max_seq_len, d_model]
         :return: Positional encodings of shape [max_seq_len, d_model]
         """
        x_shape = x.shape()
        if len(x_shape) == 2:
            seq_len, d_model = x_shape
            return x + self.encoding.data[:seq_len]
        elif len(x_shape) == 3:
            batch_size, seq_len, d_model = x_shape
            pe_expanded = self.encoding.data[:seq_len][np.newaxis, :, :]  # (1, seq_len, d_model)
            return x + pe_expanded
        else:
            raise ValueError(f"Unsupported input shape: {x_shape}")

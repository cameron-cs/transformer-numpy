import numpy as np

from src.nn import Module, Dropout
from src.tensor import Tensor


class PositionalEncodingLayer(Module):
    """
        Compute the positional encoding for a given position and dimension index
        using the sinusoidal encoding scheme from the Transformer architecture.

        Positional encoding represents a pattern that can be learned by the model

        Adds position-specific bias vectors to token embeddings.

        The sentence "I love jazz because it is smoother" was converted into embeddings by the InputEmbeddingLayer

        Token:        I        love       jazz     because     it       is      smoother
        Position:     0          1          2         3         4        5         6
                    ─────────────────────────────────────────────────────────────────────
        Dim 0 (sin)  0        0.8415     0.9093     0.1411   -0.7568  -0.9589   -0.2794
        Dim 1 (cos)  1        0.5403    -0.4161    -0.9899   -0.6536   0.2836    0.9602
        Dim 2 (sin)  0        0.0998     0.1987     0.2955    0.3894   0.4794    0.5646
        Dim 3 (cos)  1        0.9950     0.9801     0.9553    0.9210   0.8776    0.8253
        Dim 4 (sin)  0        0.00999    0.01999    0.02998   0.03996  0.04993   0.05988
        Dim 5 (cos)  1        0.9999     0.9998     0.9995    0.9992   0.9987    0.9982

        The formulas used are:
            PE(pos, 2i)   = sin(pos / (10000 ** (2i / d_model)))
            PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model)))

        This injects fixed positional information into each embedding without training. These sinusoidal patterns allow the model to learn relative positions like:
            - "because" comes after "jazz"
            - "smoother" is at the end

        It affects Q, K, and V equally — because they're computed from the input embeddings.

        PositionalEncodingLayer gives each token an identity in space.

        Parameters
        ----------
        d_model : int
            The total dimensionality of the model's embeddings (e.g., 512).

        seq_len : int
            The index of the embedding dimension. Must satisfy 0 <= i < d_model.
        """

    def __init__(self, d_model: int, seq_len: int, p_drop: float = 0.1):
        super(PositionalEncodingLayer, self).__init__()
        # create a tensor of shape (seq_len, d_model)
        # position in the sequence (e.g. token index 0, 1, 2...)
        # full encoding matrix
        pe = np.zeros((seq_len, d_model), dtype=np.float32)

        # create a tensor of shape (seq_len)
        pos: Tensor = Tensor.arange(0, seq_len).float().unsqueeze(1)  # [seq_len, 1]

        # create a 2i tensor
        # i = dimension index (ranging from 0 to (d_model / d) - 1)
        # 2i = even indices
        # 2i + 1 = odd indices
        # it indexes the even dimensions of the model's embedding vector
        # if 2i is low (0, 1, 2) - high-frequency encoding = captures local word order
        # if 2i is high (200, 240, 250) - low-frequency encoding = captures global sentence position
        _2i: Tensor = Tensor.arange(0, d_model, 2).float()  # [d_model // 2]

        # the divisor term: 10000 ** (2i / d_model)
        denominator = (10000 ** (_2i.data / d_model)).reshape(1, -1)  # [1, d_model // 2]

        # angle rates
        angle_rads = pos.data / denominator  # [seq_len, d_model // 2]

        # 0..2 -> 0, 2, 4, 6, 8, 10
        pe[:, 0::2] = np.sin(angle_rads)  # even
        # 1..2 -> 1, 3, 5, 7, 9, 11
        pe[:, 1::2] = np.cos(angle_rads)  # odd

        # constant Tensor (no gradients)
        self.pe = Tensor(pe, requires_grad=False).unsqueeze(0)  # (1, seq_len, d_model)
        self.dropout = Dropout(p_drop)

    def forward(self, x: Tensor):
        """
         Returns the positional encoding slice to be added to input embeddings.

         :param x: Tensor of shape [batch_size, max_seq_len, d_model]
         :return: Positional encodings of shape [max_seq_len, d_model]
         """
        # slice correct amount of positional encodings
        pe_sentence = Tensor(self.pe.data[:, :x.shape()[1], :])
        # don't learn this positional encoding -> it is fixed
        pe_sentence.requires_grad = False
        x = x + pe_sentence
        return self.dropout(x)


import math

from src.tensor import Tensor
from src.nn import Module, Embedding


class InputEmbeddingLayer(Module):
    """
    A token embedding layer that maps input token indices to dense vectors and scales them by sqrt(d_model).

    Attributes:
        d_model (int): Dimensionality of the output embeddings (must match model dimension).
        vocab_size (int): Number of unique tokens in the vocabulary.
        embedding (Embedding): The underlying learnable embedding lookup table.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the InputEmbeddingLayer.

        Args:
            d_model (int): The dimensionality of each embedding vector.
            vocab_size (int): Total number of tokens in the vocabulary.
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, input: Tensor) -> Tensor:
        """
        Perform a forward pass of the embedding layer.

        Args:
            input (Tensor): Tensor of token indices with shape (batch_size, seq_len) or similar.

        Returns:
            Tensor: Scaled embeddings of shape (batch_size, seq_len, d_model)
        """
        x: Tensor = self.embedding(input)  # lookup embeddings
        return x * math.sqrt(self.d_model)

import math

from src.tensor import Tensor
from src.nn import Module, Embedding


class InputEmbeddingLayer(Module):
    """
    A token embedding layer that maps input token indices to dense vectors and scales them by sqrt(d_model).

    Token embeddings give each word a vector identity that the model can optimise.

    Step-by-step:
        1. Look up each token's ID in the embedding matrix.
        2. Return the vector corresponding to that token.
        3. Scale by √d_model to normalise variance across depths.

    Example (before positional encoding):
        Sentence: "I love jazz because it is smoother"

        Token IDs:     [0, 1, 2, 3, 4, 5, 6]     # based on vocabulary
        Vocab Size:    50
        Embedding Dim: 4
        d_model:       4

        Assume initial embedding matrix (random init):
            embedding.weight = [
                [ 0.10,  0.30, -0.20,  0.50],  # "I"
                [ 0.25, -0.10,  0.40,  0.05],  # "love"
                ...
            ]

    Then InputEmbeddingLayer(input) produces:
        ┌────────────┬────────────┬────────────┬────────────┐
        │   0.20     │   0.60     │  -0.40     │   1.00     │ ← "I"
        │   0.50     │  -0.20     │   0.80     │   0.10     │ ← "love"
        │    ...     │    ...     │    ...     │    ...     │
        └────────────┴────────────┴────────────┴────────────┘
        (scaled by √4 = 2.0)

    This produces token-level vector identities that will be summed with positional encodings.
    These embeddings are learned during training to capture semantic structure.

    Attributes:
        d_model (int): Dimensionality of the output embeddings (must match model dimension).
        vocab_size (int): Number of unique tokens in the vocabulary.
        embedding (Embedding): The underlying learnable embedding lookup table.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialise the InputEmbeddingLayer.

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

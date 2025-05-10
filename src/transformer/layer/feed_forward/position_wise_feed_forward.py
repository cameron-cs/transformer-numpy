from src.tensor import Tensor
from src.nn import Module, Linear, Dropout, ReLU
from src.transformer.layer.norm.norm import LayerNorm


class PositionWiseFeedForwardLayer(Module):
    """
    Implements the position-wise feedforward sublayer used in Transformer models.

    This layer consists of:
    - LayerNorm for input normalisation
    - A two-layer MLP with a ReLU activation in between
    - Dropout after the second linear layer
    - A residual connection is added to the output

    Args:
        d_model (int): Input and output dimensionality of the model.
        d_hidden (int): Hidden layer size in the feedforward network.
        d_prob (float): Dropout probability (default: 0.1).
    """
    def __init__(self, d_model, d_hidden, d_prob=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.linear_w1 = Linear(d_model, d_hidden)
        self.linear_w2 = Linear(d_hidden, d_model)
        self.relu = ReLU()
        self.layer_norm = LayerNorm(d_model, 1e-6)
        self.dropout = Dropout(d_prob)

    def forward(self, residual: Tensor) -> Tensor:
        """
        Forward pass through the position-wise feedforward block.

        Args:
            residual (Tensor): Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        x: Tensor = self.layer_norm(residual)
        x: Tensor = self.linear_w1(x)
        x: Tensor = self.relu(x)
        x: Tensor = self.linear_w2(x)
        x: Tensor = self.dropout(x)
        return x + residual  # residual connection

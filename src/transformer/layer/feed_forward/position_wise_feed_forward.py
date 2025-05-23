from src.tensor import Tensor
from src.nn import Module, Linear, Dropout, ReLU


class PositionWiseFeedForwardLayer(Module):
    """
    Implements the position-wise feedforward sublayer used in Transformer models.

    This layer consists of:
    - NormLayer for input normalisation
    - A two-layer MLP with a ReLU activation in between
    - Dropout after the second linear layer
    - A residual connection is added to the output

    Args:
        d_model (int): Input and output dimensionality of the model.
        dff (int): Hidden layer size in the feedforward network.
        p_drop (float): Dropout probability (default: 0.1).
    """
    def __init__(self, d_model, dff, p_drop: float = 0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.linear_w1 = Linear(d_model, dff)
        self.relu = ReLU()
        self.linear_w2 = Linear(dff, d_model)
        self.dropout = Dropout(p_drop)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass through the position-wise feedforward block.

        Args:
            input (Tensor): Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, dff) --> (batch, seq_len, d_model)
        x: Tensor = self.linear_w1(input)
        x: Tensor = self.relu(x)
        x: Tensor = self.dropout(x)
        x: Tensor = self.linear_w2(x)
        return x

from src.tensor import Tensor
from src.nn import Module, Linear, Dropout, ReLU


class PositionWiseFeedForwardLayer(Module):
    """
    Applies residual connections with layer normalisation and dropout.

    This block wraps any sublayer (e.g., attention or feedforward), standardises the input,
    adds dropout regularisation, and sums the original input to the result.

       Pre-norm design:
        Normalise before sublayer for better gradient flow and stable training.

       Example (token: "jazz"):
        Let's say the self-attention or feedforward output vector for "jazz" is:
            sublayer(x) = [0.9, -1.1, 0.3, 0.5]
        The normalised input x was:
            norm(x)     = [0.5, -0.5, 0.0, 0.0]
        After dropout, we get:
            dropped     = [0.9,  0.0, 0.3, 0.5]   # some dims dropped
        Residual output:
            out         = input + dropped
                        = [0.5, -0.5, 0.0, 0.0] + [0.9, 0.0, 0.3, 0.5]
                        = [1.4, -0.5, 0.3, 0.5]

    This helps preserve input information and gradients over many layers.
    """
    def __init__(self, d_model: int, dff: int, p_drop: float = 0.1):
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

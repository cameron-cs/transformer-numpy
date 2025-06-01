from src.nn import Module, Dropout
from src.tensor import Tensor
from src.transformer.layer.norm import NormLayer


class ResidualConnectionLayer(Module):
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

    Args:
        features (int): Feature dimension, e.g., 512
        p_drop (float): Dropout rate (e.g., 0.1)
    """
    def __init__(self, features: int, p_drop: float):
        super(ResidualConnectionLayer, self).__init__()
        self.norm_layer = NormLayer(features)
        self.dropout = Dropout(p_drop)

    def forward(self, input: Tensor, sublayer: Module) -> Tensor:
        """
        Forward pass through the residual connection block.

        Args:
            input: Input tensor of shape (batch_size, sequence_len, features).
            sublayer: A callable module (e.g., attention or feedforward layer).

        Returns:
            Tensor: Output tensor of the same shape as input, with residual connection applied.

        Flow:
            1. Normalise the input.
            2. Pass normalised input through the sublayer.
            3. Apply dropout to the sublayer output.
            4. Add the original input to the dropped output (residual connection).
        """
        x: Tensor = self.norm_layer(input)
        x: Tensor = sublayer(x)
        x: Tensor = x + self.dropout(x)
        return x


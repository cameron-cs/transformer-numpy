from src.nn import Module, Dropout
from src.tensor import Tensor
from src.transformer.layer.norm import NormLayer


class ResidualConnectionBlock(Module):
    """
    Implements a residual connection around a given sublayer.
    This is a core building block used in Transformer architectures.

    Specifically:
    - Applies Layer Normalization before the sublayer ("Pre-Norm" structure).
    - Applies a Dropout after the sublayer.
    - Adds the original input to the processed output (residual connection).

    Args:
        features (int): Number of feature dimensions in the input tensor.
        dropout (float): Dropout rate applied after the sublayer output.
    """
    def __init__(self, features: int, dropout: float):
        super(ResidualConnectionBlock, self).__init__()
        self.norm_layer = NormLayer(features)
        self.dropout = Dropout(dropout)

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
        x: Tensor = sublayer.forward(x)
        x: Tensor = x + self.dropout(x)
        return x


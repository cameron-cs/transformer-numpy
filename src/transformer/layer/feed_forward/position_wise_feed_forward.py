from src.tensor import Tensor
from src.nn import Module, Linear, Dropout, ReLU


class PositionWiseFeedForwardLayer(Module):
    """
    Each position (i.e., each token vector) in the sequence is passed independently through the same two-layer
    fully connected network. This layer is shared across time and does not mix information between tokens.

    Mathematically:
        FF(x) = max(0, xW₁ + b₁)W₂ + b₂

    Where:
        - x: input tensor of shape (B, L, d_model)
        - W₁: first weight matrix of shape (d_model, d_ff)
        - b₁: first bias vector of shape (d_ff,)
        - W₂: second weight matrix of shape (d_ff, d_model)
        - b₂: second bias vector of shape (d_model,)

    ---------------------------------------------------------------------------------------------------------
    Purpose:
        - Increases model capacity by allowing non-linear transformations of each token embedding.
        - Introduces depth and flexibility into each layer of the encoder/decoder.
        - Unlike convolutions or attention, it does not share computation across positions.

    ---------------------------------------------------------------------------------------------------------
    Example:

        Sentence: ["I", "love", "jazz", "because", "it", "is", "smoother"]

        Suppose each token is represented as a 512-dim vector (d_model = 512)

        The FFN applies:
            - A linear transformation to 2048-dim (d_ff = 2048)
            - ReLU activation
            - Dropout
            - Then projects back to 512-dim

        Each token vector is transformed independently:
            x_i → Linear(512 → 2048) → ReLU → Dropout → Linear(2048 → 512)

    ---------------------------------------------------------------------------------------------------------
    Visualisation (token: "jazz"):

           [ Input ]
               ↓
        ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
        │ Linear (W₁)   │──────▶│   ReLU        │──────▶│ Linear (W₂)   │
        │ (512 → 2048)  │       │ element-wise  │       │ (2048 → 512)  │
        └───────────────┘       └───────────────┘       └───────────────┘
                                                               ↓
                                                           [ Output ]

        Output is same shape as input: (B, L, d_model)
        Transformation happens for each position separately → token-wise nonlinearity

    ---------------------------------------------------------------------------------------------------------
    Args:
        d_model (int): Dimensionality of the model (input and output size of the FFN)
        dff (int): Inner layer dimensionality (usually 2–4x d_model)
        p_drop (float): Dropout probability after activation (default: 0.1)

    Returns:
        Tensor: Output tensor of same shape as input, transformed position-wise
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

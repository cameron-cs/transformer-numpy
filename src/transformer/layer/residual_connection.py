from src.nn import Module, Dropout
from src.tensor import Tensor
from src.transformer.layer.norm import NormLayer


class ResidualConnectionBlock(Module):

    def __init__(self, features: int, dropout: float):
        super(ResidualConnectionBlock, self).__init__()
        self.norm_layer = NormLayer(features)
        self.dropout = Dropout(dropout)

    def forward(self, input: Tensor, sublayer: Module) -> Tensor:
        x: Tensor = self.norm_layer(input)
        x: Tensor = sublayer.forward(x)
        x: Tensor = x + self.dropout(x)
        return x


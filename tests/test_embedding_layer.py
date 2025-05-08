import numpy as np

from src.transformer.layer.embedding.positional_encoding import PositionalEncodingLayer
from src.tensor import Tensor


def test_positional_encoding_batched():
    batch_size = 4
    seq_len = 30
    max_seq_len = 500
    d_model = 16

    x = Tensor(np.zeros((batch_size, seq_len, d_model)))
    pe = PositionalEncodingLayer(d_model, max_seq_len)
    out = pe(x)
    assert out.shape() == (batch_size, seq_len, d_model), f"Unexpected shape: {out.shape()}"


def test_positional_encoding_value_range():
    d_model = 8
    max_seq_len = 50
    seq_len = 10

    x = Tensor(np.zeros((1, seq_len, d_model)))
    pe = PositionalEncodingLayer(d_model, max_seq_len)
    out = pe(x)

    assert np.all(out.data >= -1.0) and np.all(out.data <= 1.0), "Values out of expected range [-1, 1]"


def test_positional_encoding_determinism():
    d_model = 32
    max_seq_len = 100
    seq_len = 20

    x = Tensor(np.zeros((1, seq_len, d_model)))
    pe = PositionalEncodingLayer(d_model, max_seq_len)
    out1 = pe(x)
    out2 = pe(x)
    assert np.allclose(out1.data, out2.data), "Output is not deterministic"


if __name__ == '__main__':
    test_positional_encoding_batched()
    test_positional_encoding_value_range()
    test_positional_encoding_determinism()

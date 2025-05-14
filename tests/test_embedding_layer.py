import numpy as np

from src.transformer.layer.embedding.positional_encoding import PositionalEncodingLayer
from src.tensor import Tensor

sentence = ["I", "love", "jazz", "because", "it", "is", "smoother"]
seq_len = len(sentence)
batch_size = 1
d_model = 8


def test_positional_encoding_batched():
    batch_size = 4
    seq_len = 30
    max_seq_len = 500
    d_model = 16

    x = Tensor(np.zeros((batch_size, seq_len, d_model)))
    pe = PositionalEncodingLayer(d_model, max_seq_len)
    out = pe(x)
    assert out.shape() == (batch_size, seq_len, d_model), f"Unexpected shape: {out.shape()}"


def test_pe_shape_on_sentence():
    x = Tensor(np.zeros((batch_size, seq_len, d_model), dtype=np.float32))
    pe_layer = PositionalEncodingLayer(d_model, seq_len, p_drop=0.0)
    out = pe_layer(x)
    assert out.shape() == (batch_size, seq_len, d_model), \
        f"Output shape mismatch: got {out.shape()}, expected {(batch_size, seq_len, d_model)}"


def test_pe_determinism_on_sentence():
    x1 = Tensor(np.zeros((batch_size, seq_len, d_model), dtype=np.float32))
    x2 = Tensor(np.zeros((batch_size, seq_len, d_model), dtype=np.float32))
    pe_layer = PositionalEncodingLayer(d_model, seq_len, p_drop=0.0)
    out1 = pe_layer(x1)
    out2 = pe_layer(x2)
    np.testing.assert_allclose(out1.data, out2.data, atol=1e-6, err_msg="PE should be deterministic")


def test_pe_sin_cos_structure_on_sentence():
    pe_layer = PositionalEncodingLayer(d_model, seq_len, p_drop=0.0)
    pe_vector = pe_layer.pe.data[0, 0]  # first token "I"
    sin_part = pe_vector[::2]
    cos_part = pe_vector[1::2]

    assert np.all(np.abs(sin_part) <= 1.0), "Sin part should be in [-1, 1]"
    assert np.all(np.abs(cos_part) <= 1.0), "Cos part should be in [-1, 1]"


def test_dropout_preserves_shape_on_sentence():
    x = Tensor(np.zeros((batch_size, seq_len, d_model), dtype=np.float32))
    pe_layer = PositionalEncodingLayer(d_model, seq_len, p_drop=0.3)
    out = pe_layer(x)
    assert out.shape() == (batch_size, seq_len, d_model), "Dropout should not change shape"


def test_positional_encoding_visualization():
    pe = PositionalEncodingLayer(d_model=d_model, seq_len=seq_len, p_drop=0.0)

    # (1, seq_len, d_model) → (seq_len, d_model)
    pe_matrix = pe.pe.data.squeeze()

    assert np.isclose(pe_matrix[1, 0], 0.8415, atol=1e-4)
    assert np.isclose(pe_matrix[2, 1], -0.4161, atol=1e-4)


if __name__ == '__main__':
    test_positional_encoding_batched()
    test_pe_shape_on_sentence()
    test_pe_shape_on_sentence()
    test_pe_determinism_on_sentence()
    test_pe_sin_cos_structure_on_sentence()
    test_dropout_preserves_shape_on_sentence()
    test_positional_encoding_visualization()

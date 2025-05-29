import math

import numpy as np

from src.transformer.layer.embedding.input import InputEmbeddingLayer
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


def test_input_embedding_shape_and_scale():
    vocab_size = 10
    d_model = 4
    batch_size = 2
    seq_len = 3

    # token indices
    input_ids = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=False)

    # init embedding layer
    layer = InputEmbeddingLayer(d_model=d_model, vocab_size=vocab_size)

    # known weights for deterministic test
    layer.embedding.weight.data[:] = np.arange(vocab_size * d_model).reshape(vocab_size, d_model)

    # forward pass
    output = layer.forward(input_ids)

    # shape
    assert output.shape() == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape()}"

    expected_scale = math.sqrt(d_model)
    raw_vector = layer.embedding.weight.data[1]
    scaled_vector = output.data[0, 0]
    assert np.allclose(scaled_vector, raw_vector * expected_scale), \
        f"Expected scaled vector {raw_vector * expected_scale}, got {scaled_vector}"


def test_input_embedding_reproducibility():
    vocab_size = 20
    d_model = 8

    input_ids = Tensor(np.array([[5, 5, 5]]), requires_grad=False)
    layer = InputEmbeddingLayer(d_model=d_model, vocab_size=vocab_size)

    out = layer(input_ids)

    first = out.data[0, 0]
    second = out.data[0, 1]
    third = out.data[0, 2]

    assert np.allclose(first, second), "Embeddings for same token index must match"
    assert np.allclose(second, third), "Embeddings for same token index must match"


if __name__ == '__main__':
    test_positional_encoding_batched()
    test_pe_shape_on_sentence()
    test_pe_shape_on_sentence()
    test_pe_determinism_on_sentence()
    test_pe_sin_cos_structure_on_sentence()
    test_dropout_preserves_shape_on_sentence()
    test_input_embedding_shape_and_scale()
    test_input_embedding_reproducibility()

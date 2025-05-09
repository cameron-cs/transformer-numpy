import math

import numpy as np

from src.tensor import Tensor
from src.transformer.blocks.attention.multi_head_attention import MultiHeadAttentionBlock
from src.transformer.blocks.attention.scaled_dot_product import ScaledDotProductAttentionBlock

vocab = {
    "the": [1.0, 0.0],
    "cat": [0.0, 1.0],
    "sat": [0.5, 0.5],
    "on": [0.2, 0.8],
    "mat": [0.0, 1.0],
}


def encode(text, vocab):
    tokens = text.split()
    vectors = [vocab[t] for t in tokens]
    return Tensor(np.array(vectors).reshape(1, 1, len(tokens), -1))  # (1, 1, L, d)


def test_scaled_dot_product_attention():
    q = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True)  # shape (1, 1, 2, 2)
    k = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True)
    v = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

    attention = ScaledDotProductAttentionBlock()
    out, score = attention(q, k, v)

    # score shape
    assert score.shape() == (1, 1, 2, 2), f"Expected score shape (1, 1, 2, 2), got {score.shape}"
    # output shape
    assert out.shape() == (1, 1, 2, 2), f"Expected output shape (1, 1, 2, 2), got {out.shape}"

    # sanity check values (manually or approximately)
    d_k = q.size(-1)
    expected_raw_scores = (q.data @ k.data.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
    expected_softmax = np.exp(expected_raw_scores) / np.sum(np.exp(expected_raw_scores), axis=-1, keepdims=True)
    expected_out = expected_softmax @ v.data

    assert np.allclose(score.data, expected_softmax, atol=1e-5), "Softmax scores incorrect"
    assert np.allclose(out.data, expected_out, atol=1e-5), "Attention output incorrect"

    # backward check
    out.sum().backward()
    assert q.grad is not None, "q.grad is None"
    assert k.grad is not None, "k.grad is None"
    assert v.grad is not None, "v.grad is None"


def test_multihead_attention_output_shape():
    batch_size, seq_len, d_model, h = 2, 5, 8, 2
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, dropout_p=0.0)
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    out = mha(x, x, x)  # Self-attention
    assert out.data.shape == (
        batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {out.data.shape}"


def test_multihead_attention_split_concat_inverse():
    batch_size, seq_len, d_model, h = 2, 6, 8, 4
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model)
    x = Tensor(np.random.randn(batch_size, seq_len, d_model))

    split = mha.split(x)
    recon = mha.concat(split)
    assert np.allclose(x.data, recon.data), "Split followed by concat should recover original input"


def test_multihead_attention_forward_deterministic():
    np.random.seed(42)
    batch_size, seq_len, d_model, h = 1, 3, 4, 2
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, dropout_p=0.0)

    x = Tensor(np.ones((batch_size, seq_len, d_model)), requires_grad=True)
    out1 = mha(x, x, x)
    out2 = mha(x, x, x)
    assert np.allclose(out1.data, out2.data), "Deterministic forward failed"


def test_multihead_attention_with_mask():
    batch_size, seq_len, d_model, h = 1, 4, 8, 2
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, dropout_p=0.0)

    x = Tensor(np.random.randn(batch_size, seq_len, d_model))
    mask = np.array([[0, 0, -np.inf, -np.inf]])
    mask = mask.reshape(1, 1, 1, -1)  # broadcast over heads

    out_masked = mha(x, x, x, mask=mask)
    out_unmasked = mha(x, x, x, mask=None)

    assert not np.allclose(out_masked.data, out_unmasked.data), "Mask did not affect output"


def test_multihead_attention_backward():
    batch_size, seq_len, d_model, h = 2, 5, 8, 4
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, dropout_p=0.0)

    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    out = mha(x, x, x)
    out.mean().backward()

    assert x.grad is not None, "Gradient not propagated back"
    assert x.grad.shape == x.data.shape, "Incorrect gradient shape"


def test_attention_cat_attends_to_mat():
    # 1: fixed vocab
    vocab = {
        "the": Tensor([[[[1.0, 0.0]]]]),  # (B=1, H=1, L=1, D=2)
        "cat": Tensor([[[[0.0, 1.0]]]]),  # (B=1, H=1, L=1, D=2)
        "sat": Tensor([[[[0.5, 0.5]]]]),
        "on": Tensor([[[[0.2, 0.8]]]]),
        "mat": Tensor([[[[0.0, 1.0]]]]),  # (B=1, H=1, L=1, D=2)
    }

    # 2: encode the sentence into q, k, v tensors using Tensor API
    sentence = ["the", "cat", "sat", "on", "mat"]
    qkv_data = [vocab[token].data[0, 0, 0] for token in sentence]

    # the qkv tensor using Tensor API (shape: (B=1, H=1, L=5, D=2))
    qkv_tensor = Tensor(np.array(qkv_data).reshape(1, 1, 5, 2))  # Tensor wrapping np.array
    q = k = v = qkv_tensor

    # 3: attention
    attn = ScaledDotProductAttentionBlock()
    out, score = attn(q, k, v)

    # 4: extract attention weights for the second word "cat" (index 1)
    cat_weights = score.data[0, 0, 1]  # Attention weights for "cat"

    # 5: assertions based on expected attention pattern
    assert np.isclose(cat_weights[1], cat_weights[4]), "Cat and Mat should have equal attention"
    assert cat_weights[1] > cat_weights[2], "Cat should attend more to itself than to 'sat'"
    assert cat_weights[1] > cat_weights[0], "Cat should attend more to itself than to 'the'"
    assert np.isclose(cat_weights.sum(), 1.0), "Attention weights should sum to 1"


if __name__ == '__main__':
    test_scaled_dot_product_attention()
    test_multihead_attention_output_shape()
    test_multihead_attention_split_concat_inverse()
    test_multihead_attention_forward_deterministic()
    test_multihead_attention_backward()
    test_attention_cat_attends_to_mat()

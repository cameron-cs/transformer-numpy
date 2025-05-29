import math

import numpy as np

from src import nn
from src.nn import Embedding
from src.tensor import Tensor
from src.transformer.layer.attention.multi_head_attention import MultiHeadAttentionBlock
from src.transformer.layer.attention.scaled_dot_product import ScaledDotProductAttentionBlock
from src.transformer.layer.embedding.positional_encoding import PositionalEncodingLayer


def test_scaled_dot_product_attention():
    q = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True)  # shape (1, 1, 2, 2)
    k = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True)
    v = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

    attention = ScaledDotProductAttentionBlock(dropout=0.1)
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
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, p_drop=0.0)
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    out = mha(x, x, x)  # Self-attention
    assert out.data.shape == (
        batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {out.data.shape}"


def test_multihead_attention_split_concat_inverse():
    batch_size, seq_len, d_model, h = 2, 6, 8, 4
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, p_drop=0.1)
    x = Tensor(np.random.randn(batch_size, seq_len, d_model))

    split = mha.split(x)
    recon = mha.concat(split)
    assert np.allclose(x.data, recon.data), "Split followed by concat should recover original input"


def test_multihead_attention_forward_deterministic():
    np.random.seed(42)
    batch_size, seq_len, d_model, h = 1, 3, 4, 2
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, p_drop=0.0)

    x = Tensor(np.ones((batch_size, seq_len, d_model)), requires_grad=True)
    out1 = mha(x, x, x)
    out2 = mha(x, x, x)
    assert np.allclose(out1.data, out2.data), "Deterministic forward failed"


def test_multihead_attention_with_mask():
    batch_size, seq_len, d_model, h = 1, 4, 8, 2
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, p_drop=0.0)

    x = Tensor(np.random.randn(batch_size, seq_len, d_model))
    mask = np.array([[0, 0, -np.inf, -np.inf]])
    mask = mask.reshape(1, 1, 1, -1)  # broadcast over heads

    out_masked = mha(x, x, x, mask=mask)
    out_unmasked = mha(x, x, x, mask=None)

    assert not np.allclose(out_masked.data, out_unmasked.data), "Mask did not affect output"


def test_multihead_attention_backward():
    batch_size, seq_len, d_model, h = 2, 5, 8, 4
    mha = MultiHeadAttentionBlock(h=h, d_model=d_model, p_drop=0.0)

    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    out = mha(x, x, x)
    out.mean().backward()

    assert x.grad is not None, "Gradient not propagated back"
    assert x.grad.shape == x.data.shape, "Incorrect gradient shape"


def test_mha_cat_equals_mat_attention():
    vocab = {
        "the": Tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
        "cat": Tensor([[[[0.0, 1.0, 0.0, 0.0]]]]),
        "sat": Tensor([[[[0.0, 0.5, 0.5, 0.0]]]]),
        "on": Tensor([[[[0.0, 0.2, 0.8, 0.0]]]]),
        "mat": Tensor([[[[0.0, 1.0, 0.0, 0.0]]]]),
    }

    sentence = ["the", "cat", "sat", "on", "mat"]
    x_np = np.stack([vocab[t].data[0, 0, 0] for t in sentence])  # (5, 4)
    x = Tensor(x_np[None, :, :])  # (1, 5, 4)

    mha = MultiHeadAttentionBlock(h=2, d_model=4)

    # run attention
    out = mha(x, x, x)

    # manually inspect weights
    q = mha.split(mha.linear_wq(x))
    k = mha.split(mha.linear_wk(x))
    v = mha.split(mha.linear_wv(x))
    _, attn = mha.attention(q, k, v)

    attn_weights = attn.data[0, 0, 1]

    assert abs(attn_weights[1] - attn_weights[4]) < 1e-4, "'cat' and 'mat' should receive equal attention"
    assert np.isclose(attn_weights.sum(), 1.0), "Attention distribution must sum to 1"

    # identical outputs for "cat" and "mat"
    assert np.allclose(out.data[0, 1], out.data[0, 4], atol=1e-4), "Output vectors should match for 'cat' and 'mat'"


def test_mha_attention_semantics_and_gradients():
    vocab = {
        "dogs": Tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
        "bark": Tensor([[[[0.0, 1.0, 0.0, 0.0]]]]),
        "cats": Tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
        "meow": Tensor([[[[0.0, 1.0, 0.0, 0.0]]]]),
        "loud": Tensor([[[[0.0, 0.5, 0.5, 0.0]]]]),
        "quiet": Tensor([[[[0.0, 0.5, 0.5, 0.0]]]]),
    }

    sentence = ["dogs", "bark", "loud", "cats", "meow", "quiet"]
    x_np = np.stack([vocab[t].data[0, 0, 0] for t in sentence])  # (6, 4)
    x = Tensor(x_np[None, :, :], requires_grad=True)  # (1, 6, 4)

    mha = MultiHeadAttentionBlock(h=2, d_model=4, p_drop=0.1)

    class IdentityLinear(nn.Module):
        def forward(self, x): return x

    mha.linear_wq = IdentityLinear()
    mha.linear_wk = IdentityLinear()
    mha.linear_wv = IdentityLinear()

    out = mha(x, x, x)

    # ==== attention similarity ====
    q = mha.split(x)
    k = mha.split(x)
    v = mha.split(x)
    _, attn = mha.attention(q, k, v)

    attn_dogs = attn.data[0, 0, 0]  # head 0, position 0 ("dogs")
    attn_cats = attn.data[0, 0, 3]  # head 0, position 3 ("cats")

    # dogs and cats have identical embeddings → expect similar attention patterns
    np.testing.assert_allclose(attn_dogs, attn_cats, atol=1e-4)

    # sanity: attention scores are normalised
    assert np.isclose(attn_dogs.sum(), 1.0)
    assert np.isclose(attn_cats.sum(), 1.0)

    # ==== gradient propagation ====
    # sum output and backward
    out.sum().backward()

    # gradients flowed through input
    assert x.grad is not None, "Input tensor must receive gradient"
    assert np.any(x.grad.data != 0), "Gradient through MHA must not be zero"


def test_mha_semantic_attention_and_gradient_check():
    # semantic vocab with embedding dim 4
    vocab = {
        "king": Tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
        "queen": Tensor([[[[0.9, 0.1, 0.0, 0.0]]]]),
        "sat": Tensor([[[[0.0, 1.0, 0.0, 0.0]]]]),
        "on": Tensor([[[[0.0, 0.9, 0.1, 0.0]]]]),
        "throne": Tensor([[[[0.0, 0.8, 0.2, 0.0]]]]),
        "wearing": Tensor([[[[0.0, 0.0, 1.0, 0.0]]]]),
        "golden": Tensor([[[[0.0, 0.0, 0.9, 0.1]]]]),
        "crown": Tensor([[[[0.0, 0.0, 0.8, 0.2]]]]),
    }

    sentence = ["king", "queen", "sat", "on", "throne", "wearing", "golden", "crown"]
    x_np = np.stack([vocab[t].data[0, 0, 0] for t in sentence])  # (8, 4)
    x = Tensor(x_np[None, :, :])  # (1, 8, 4)
    x.requires_grad = True

    mha = MultiHeadAttentionBlock(h=2, d_model=4, p_drop=0.1)

    # === force identity projections for Q, K, V ===
    identity_matrix = np.eye(4)
    for linear in [mha.linear_wq, mha.linear_wk, mha.linear_wv]:
        linear.weights.data[:] = identity_matrix
        linear.bias.data[:] = np.zeros_like(linear.bias.data)

    # === forward and backward ===
    out = mha(x, x, x)
    out.sum().backward()

    # === attention weights ===
    q = mha.split(mha.linear_wq(x))
    k = mha.split(mha.linear_wk(x))
    v = mha.split(mha.linear_wv(x))
    _, attn = mha.attention(q, k, v)

    idx = sentence.index("queen")
    attn_weights = attn.data[0, 0, idx]

    print("\nAttention from 'queen':")
    for token, weight in zip(sentence, attn_weights):
        print(f"{token:>8}: {weight:.3f}")

    assert attn_weights[sentence.index("king")] > 0.15
    assert attn_weights[sentence.index("crown")] > 0.1

    # === gumerical grad check ===
    epsilon = 1e-4
    analytical_grad = np.array(x.grad.data)
    numerical_grad = np.zeros_like(x.data)

    for i in range(x.data.shape[1]):
        for j in range(x.data.shape[2]):
            orig = x.data[0, i, j]

            x.data[0, i, j] = orig + epsilon
            out_plus = mha(x, x, x).sum().data

            x.data[0, i, j] = orig - epsilon
            out_minus = mha(x, x, x).sum().data

            num_grad = (out_plus - out_minus) / (2 * epsilon)
            numerical_grad[0, i, j] = num_grad

            x.data[0, i, j] = orig

    diff = np.abs(numerical_grad - analytical_grad)
    max_diff = diff.max()

    assert max_diff < 1e-2, f"Gradient check failed! Max diff too large: {max_diff}"


tokens = ["I", "love", "jazz", "because", "it", "is", "smoother"]
B, H, L, d_k, d_v = 1, 1, len(tokens), 32, 32  # single batch, single head


def embed_tokens(seed=0):
    np.random.seed(seed)
    return (
        Tensor(np.random.randn(B, H, L, d_k)),
        Tensor(np.random.randn(B, H, L, d_k)),
        Tensor(np.random.randn(B, H, L, d_v)),
    )


def test_attention_shape_real_tokens():
    q, k, v = embed_tokens()
    attn = ScaledDotProductAttentionBlock(dropout=0.0)
    out, score = attn(q, k, v)
    assert out.shape() == (B, H, L, d_v)
    assert score.shape() == (B, H, L, L)


def test_softmax_rows_sum_to_1_real_tokens():
    q, k, v = embed_tokens()
    attn = ScaledDotProductAttentionBlock(dropout=0.0)
    _, score = attn(q, k, v)
    sums = score.data.sum(axis=-1)  # shape: (B, H, L)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-6)



def test_dropout_applied_real_tokens():
    """
    Check dropout integration doesn't crash and keeps shape.
    """
    q, k, v = embed_tokens()
    attn = ScaledDotProductAttentionBlock(dropout=0.5)
    out, score = attn(q, k, v)

    assert out.shape() == (B, H, L, d_v)
    assert score.shape() == (B, H, L, L)


def test_token_attention_weights_example():
    """
    expected attention behavior on toy sentence:
    ["I", "love", "jazz", "because", "it", "is", "smoother"]

    Checks that:
    - "love" attends significantly to "jazz"
    - "because" attends more to "I" and "love" than "smoother"
    - "jazz" attends significantly to "smoother"
    """
    q, k, v = embed_tokens(seed=42)
    attn = ScaledDotProductAttentionBlock(dropout=0.0)
    _, score = attn(q, k, v)

    # select attention matrix: shape (B=1, H=1, L_q=7, L_k=7)
    weights = score.data[0, 0]  # shape: (7, 7)

    # index lookup
    idx = {t: i for i, t in enumerate(tokens)}

    # assert: "love" → "jazz"
    love_to_jazz = weights[idx["love"], idx["jazz"]]
    assert love_to_jazz > 0.10, f'"love" → "jazz" attention too low: {love_to_jazz:.2f}'

    # assert: "because" → "I" and "love"
    because_to_I = weights[idx["because"], idx["I"]]
    because_to_love = weights[idx["because"], idx["love"]]
    because_to_smoother = weights[idx["because"], idx["smoother"]]

    assert because_to_I > because_to_smoother, f'"because" should attend more to "I" than "smoother" ({because_to_I:.2f} vs {because_to_smoother:.2f})'
    assert because_to_love > because_to_smoother, f'"because" should attend more to "love" than "smoother" ({because_to_love:.2f} vs {because_to_smoother:.2f})'

    # assert: "jazz" → "smoother"
    jazz_to_smoother = weights[idx["jazz"], idx["smoother"]]
    assert jazz_to_smoother > 0.30, f'"jazz" → "smoother" attention too low: {jazz_to_smoother:.2f}'


if __name__ == '__main__':
    test_attention_shape_real_tokens()
    test_softmax_rows_sum_to_1_real_tokens()
    test_dropout_applied_real_tokens()
    test_token_attention_weights_example()
    test_multihead_attention_output_shape()
    test_multihead_attention_split_concat_inverse()
    test_multihead_attention_forward_deterministic()
    test_multihead_attention_backward()
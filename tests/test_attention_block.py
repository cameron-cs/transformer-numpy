import math

import numpy as np

from src import nn
from src.tensor import Tensor
from src.transformer.blocks.attention.multi_head_attention import MultiHeadAttentionBlock
from src.transformer.blocks.attention.scaled_dot_product import ScaledDotProductAttentionBlock


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
    qkv_tensor = Tensor(np.array(qkv_data)).reshape(1, 1, 5, 2)
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

    mha = MultiHeadAttentionBlock(h=2, d_model=4)

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

    mha = MultiHeadAttentionBlock(h=2, d_model=4)

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


if __name__ == '__main__':
    test_scaled_dot_product_attention()
    test_multihead_attention_output_shape()
    test_multihead_attention_split_concat_inverse()
    test_multihead_attention_forward_deterministic()
    test_multihead_attention_backward()
    test_attention_cat_attends_to_mat()
    test_mha_cat_equals_mat_attention()
    test_mha_attention_semantics_and_gradients()
    test_mha_semantic_attention_and_gradient_check()

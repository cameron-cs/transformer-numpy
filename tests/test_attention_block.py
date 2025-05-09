import math

import numpy as np

from src.tensor import Tensor
from src.transformer.blocks.attention.scale_dot_product import ScaleDotProductAttentionBlock


def test_scaled_dot_product_attention():

    q = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True)  # shape (1, 1, 2, 2)
    k = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True)
    v = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

    attention = ScaleDotProductAttentionBlock()
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


if __name__ == '__main__':
    test_scaled_dot_product_attention()
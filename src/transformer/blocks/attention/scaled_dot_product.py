import math

from src.nn import Module, Softmax
from src.tensor import Tensor


class ScaledDotProductAttentionBlock(Module):
    """
    Implements the **Scaled Dot-Product Attention** mechanism used as the core of the Transformer architecture.

    This block computes attention weights and applies them to the value tensor, following the equation:

        Attention(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V

    Where:
        - Q: query tensor, shape (B, H, L_q, d_k)
        - K: key tensor, shape (B, H, L_k, d_k)
        - V: value tensor, shape (B, H, L_k, d_v)
        - B: batch size
        - H: number of heads
        - L_q: query length
        - L_k: key/value length
        - d_k: key dimension
        - d_v: value dimension

    This implementation supports optional masking, and is compatible with your custom autograd-enabled `Tensor` class.
    """
    def __init__(self):
        super(ScaledDotProductAttentionBlock, self).__init__()
        self.softmax = Softmax()

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', mask=None, e=1e-12):
        """
        Performs the forward pass of scaled dot-product attention.

        Args:
            q (Tensor): Query tensor of shape (B, H, L_q, d_k)
            k (Tensor): Key tensor of shape (B, H, L_k, d_k)
            v (Tensor): Value tensor of shape (B, H, L_k, d_v)
            mask (Tensor, optional): Binary mask tensor broadcastable to (B, H, L_q, L_k),
                                     where `0` means masked (ignored), and `1` means active.
            e (float): Small constant for numerical stability (not currently used).

        Returns:
            Tuple[Tensor, Tensor]:
                - output: (B, H, L_q, d_v), the result of attention computation.
                - score:  (B, H, L_q, L_k), the attention weights after softmax.
        """
        d_tensor = q.size(-1)
        # === 1. transpose key tensor for dot product ===
        # Kᵀ: transpose last two dims to prepare for batch matmul
        kT: Tensor = k.transpose(2, 3)  # (B, H, d_k, L_k)

        # === 2. compute raw attention scores ===
        # Q @ Kᵀ: similarity between each query and key, scaled
        qkT: Tensor = q @ kT  # shape: (B, H, L_q, L_k)
        Vdk: float = math.sqrt(d_tensor)  # scale factor √d_k

        score: Tensor = qkT / Vdk  # scaled dot product scores

        # === 3. optional mask ===
        if mask:
            # set masked positions to -inf before softmax
            score = score.masked_fill(mask == 0, float('-inf'))

        # === 4. normalise scores via softmax ===
        # converts scores into probabilities (attention weights)
        score = self.softmax(score)

        # === 5. weighted sum of values ===
        # multiply attention weights with value vectors
        v: Tensor = score @ v  # shape: (B, H, L_q, d_v)
        return v, score

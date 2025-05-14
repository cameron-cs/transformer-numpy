import math

from src.nn import Module, Softmax, Dropout
from src.tensor import Tensor


class ScaledDotProductAttentionBlock(Module):
    """
    Implements the **Scaled Dot-Product Attention** mechanism used as the core of the Transformer architecture.

    Attention maps a query to a set of key-value pairs.
    This determines which source elements to focus on during decoding.

    Example sentence:
        Target: ["I", "love", "jazz", "because", "it", "is", "smoother"]

    Structure:
        PRON    VERB     NOUN     CONJ     PRON    VERB   ADJ
         I     love     jazz   because     it      is   smoother

    -------------------------------------------------------------------------------------------------------------------
        I        love         jazz                               because      it           is        smoother
    [ Query1 ]  [ Query2 ]  [ Query3 ]                         [ Query4 ]  [ Query5 ]  [ Query6 ]  [ Query7 ]
                     .                                                                                 *
                     .                                                                                 *
             * * * * .* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
             *       .   *                  *                               *             *            *              *
        . . .* . . . . . *. . . . . . . . . *. . . . . . . . . . . . . . . .*. . . . . .  * . . . . . .* . . . . .    *
        .    *       .   *             .    *                        .      *       .     *      .     *         .    *
        .    *       .   *             .    *                        .      *       .     *      .     *         .    *
       (w1) (w1)   (w2) (w2)          (w3) (w3)                     (w4)   (w4)    (w5)  (w5)   (w6)  (w6)      (w7) (w7)
        .    *       .   *             .    *                        .       *      .     *      .     *         .    *
        .    *       .   *             .    *                        .       *      .     *      .     *         .    *
        .    *       .   *             .    *                        .       *      .     *      .     *         .    *
      [ Key 1 ]    [ Key 2 ]          [ Key 3 ]                     [  Key 4  ]    [ Key 5 ]   [  Key 6  ]      [  Key 7 ]
          |           |                   |                              |             |            |               |
          |           |                   |                              |             |            |               |
          |           |                   |                              |             |            |               |
      [ Value 1 ] [ Value 2 ]         [ Value 3 ]                   [  Value 4 ]   [ Value 5 ]  [ Value 6  ]   [  Value 7 ]
        I           love                 jazz                         because          it           is           smoother

    - Each Query compares against all Keys → producing weights (w₁...w₇)
    - These weights are used to compute a weighted sum over the Values,
    - resulting in contextualised outputs for each Query token.

    Attention weights (w₁ to w₇) represent how much each word contributes:

    Query2 ("love") flows through the attention mechanism as follows:

        - it is compared against all Keys → producing similarity scores
        - these scores are normalized via softmax → giving attention weights w₁ to w₇
        - each weight wⱼ indicates how much "love" attends to word j
        - these are used to compute:
            Output("love") = w₁ · Value1 + w₂ · Value2 + ... + w₇ · Value7

    - Query2 ("love") is aligned vertically over all Key positions
    - Weights (w₁ to w₇) are shown directly below each Key
    - Final vector is a contextualised embedding for "love"

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
    """
    def __init__(self, dropout: float):
        super(ScaledDotProductAttentionBlock, self).__init__()
        self.softmax = Softmax()
        self.dropout = Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None, e=1e-12):
        """
        Performs the forward pass of scaled dot-product attention.

        Args:
            q (Tensor): Query tensor of shape (B, H, L_q, d_k)
            k (Tensor): Key tensor of shape (B, H, L_k, d_k)
            v (Tensor): Value tensor of shape (B, H, L_k, d_v)
            mask (Tensor, optional): Binary mask tensor broadcastable to (B, H, L_q, L_k),
                                     where `0` means masked (ignored), and `1` means active.
            e (float): Small constant for numerical stability.

        Returns:
            Tuple[Tensor, Tensor]:
                - output: (B, H, L_q, d_v), the result of attention computation.
                - score:  (B, H, L_q, L_k), the attention weights after softmax.
        """
        # the last dimension of Q, K, V
        d_k = q.shape()[-1]
        # === 1. transpose key tensor for dot product ===
        # Kᵀ: transpose last two dims to prepare for batch matmul
        kT: Tensor = k.transpose(2, 3)  # (B, H, d_k, L_k)

        # === 2. compute raw attention scores ===
        # Q @ Kᵀ: similarity between each query and key, scaled
        qkT: Tensor = q @ kT  # shape: (B, H, L_q, L_k)
        Vd_k: float = math.sqrt(d_k)  # scale factor √d_k

        score: Tensor = qkT / Vd_k  # scaled dot product scores

        # === 3. optional mask ===
        if mask is not None:
            # set masked positions to -inf before softmax
            score = score.masked_fill(mask == 0, float('-inf'))

        # === 4. normalise scores via softmax and apply dropout===
        # converts scores into probabilities (attention weights)
        score: Tensor = self.softmax(score)
        score: Tensor = self.dropout(score)

        # === 5. weighted sum of values ===
        # multiply attention weights with value vectors
        v: Tensor = score @ v  # shape: (B, H, L_q, d_v)
        return v, score

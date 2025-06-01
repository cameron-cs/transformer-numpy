from src.nn import Module, ModuleList
from src.tensor import Tensor
from src.transformer.layer.attention.multi_head_attention import MultiHeadAttentionBlock
from src.transformer.layer.feed_forward.position_wise_feed_forward import PositionWiseFeedForwardLayer
from src.transformer.layer.norm import NormLayer
from src.transformer.layer.residual_connection import ResidualConnectionLayer


class EncoderBlock(Module):
    """
    A full encoder layer block from the original Transformer ("Attention Is All You Need").

    Consists of:
    ┌─────────────────────────────┐
    │ 1. MultiHeadAttentionBlock  │ ← self-attention on input sequence
    ├─────────────────────────────┤
    │ 2. ResidualConnectionLayer  │ ← skip connection + dropout + LayerNorm
    ├─────────────────────────────┤
    │ 3. PositionWiseFeedForward  │ ← per-token fully connected MLP
    ├─────────────────────────────┤
    │ 4. ResidualConnectionLayer  │ ← again
    └─────────────────────────────┘

    ───────────────────────────────────────────────────────────────────────────────
    Example:
        Sentence: ["I", "love", "jazz", "because", "it", "is", "smoother"]

        Each token was embedded (token embedding + positional encoding) into vector space:
        → [ E("I") + PE[0], E("love") + PE[1], ..., E("smoother") + PE[6] ]

    ───────────────────────────────────────────────────────────────────────────────
    Stage 1: Multi-Head Self Attention

    MultiHeadAttentionBlock allows the model to jointly attend to information from
    different representation subspaces — at different positions — in parallel.

    Under the hood, it performs:
        1. Linear projections of the input to get Q, K, V for each head
        2. Applies ScaledDotProductAttention to each head in parallel
        3. Concatenates all outputs and projects them back with a linear layer

    ───────────────────────────────────────────────────────────────────────────────
    Let’s walk through the same sentence again:

        Input sentence:
            ["I", "love", "jazz", "because", "it", "is", "smoother"]

    Step 1: Linear Projections
        Q = input @ Wq,  K = input @ Wk,  V = input @ Wv
        These encode query, key, and value signals for each token

    Step 2: Split into Multiple Heads
        Each head processes different learned subspaces
        E.g. head1 captures short-range dependencies, head2 learns syntax...

    Step 3: Apply Scaled Dot-Product Attention
        For each head: softmax(Q @ K^T / sqrt(d_k)) @ V
        Then outputs are concatenated and projected → contextual embeddings

    ───────────────────────────────────────────────────────────────────────────────
    Stage 2: Residual Connection + LayerNorm

    ResidualConnectionLayer applies:
        output = x + Dropout(Sublayer(LayerNorm(x)))

    This stabilises training and preserves signal flow.
    It is applied around both MHA and FFN components.

    ───────────────────────────────────────────────────────────────────────────────
    Stage 3: Position-Wise FeedForward Network

    PositionWiseFeedForwardLayer applies:
        FF(x) = max(0, xW₁ + b₁)W₂ + b₂ (same for each token independently)

    It enhances the model’s ability to learn local transformations
    such as re-weighting attention features and introducing non-linearity.

    ───────────────────────────────────────────────────────────────────────────────
    Stage 4: Second Residual Connection + LayerNorm

    Another ResidualConnectionLayer is applied around the FFN.

    Final output is ready to be passed to the next encoder block (if any).
    """
    def __init__(self, features: int, mha_block: MultiHeadAttentionBlock, ff_block: PositionWiseFeedForwardLayer, dropout: float):
        super(EncoderBlock, self).__init__()
        self.mha_block = mha_block
        self.ff_block = ff_block
        self.res_connections = ModuleList([
            ResidualConnectionLayer(features=features, p_drop=dropout),
            ResidualConnectionLayer(features=features, p_drop=dropout)
        ])

    def forward(self, input: Tensor, src_mask) -> Tensor:
        # the first skip connection is between `x` and inout itself (Q, K, V), e.g. the sentence looking at itself
        x = self.res_connections[0](input, lambda x: self.mha_block(q=x, k=x, v=x, mask=src_mask))
        x = self.res_connections[1](x, self.ff_block)
        return x


class EncoderLayer(Module):
    """
    A full encoder layer block from the original Transformer ("Attention Is All You Need").

    Consists of:
    ┌─────────────────────────────┐
    │ 1. MultiHeadAttentionBlock  │ ← self-attention on input sequence
    ├─────────────────────────────┤
    │ 2. ResidualConnectionLayer  │ ← skip connection + dropout + LayerNorm
    ├─────────────────────────────┤
    │ 3. PositionWiseFeedForward  │ ← per-token fully connected MLP
    ├─────────────────────────────┤
    │ 4. ResidualConnectionLayer  │ ← again
    └─────────────────────────────┘

    ───────────────────────────────────────────────────────────────────────────────
    Example:
        Sentence: ["I", "love", "jazz", "because", "it", "is", "smoother"]

        Each token was embedded (token embedding + positional encoding) into vector space:
        → [ E("I") + PE[0], E("love") + PE[1], ..., E("smoother") + PE[6] ]

    ───────────────────────────────────────────────────────────────────────────────
    Stage 1: Multi-Head Self Attention

    MultiHeadAttentionBlock allows the model to jointly attend to information from
    different representation subspaces — at different positions — in parallel.

    Under the hood, it performs:
        1. Linear projections of the input to get Q, K, V for each head
        2. Applies ScaledDotProductAttention to each head in parallel
        3. Concatenates all outputs and projects them back with a linear layer


    ───────────────────────────────────────────────────────────────────────────────
    Let’s walk through the same sentence again:

        Input sentence:
            ["I", "love", "jazz", "because", "it", "is", "smoother"]


    ───────────────────────────────────────────────────────────────────────────────
    Step 1: Linear Projections

        - Q = input @ Wq  → (batch, seq_len, d_model)
        - K = input @ Wk
        - V = input @ Wv

        These matrices encode:
            • Q["love"]: how "love" will query other tokens
            • K["jazz"]: how other tokens will attend to "jazz"
            • V["it"]: value vector carrying semantic content


                      . . . .  [ Q ]          @     [ Wq ]        =     [  Q' ]
                      .    (seq, d_model)       (seq, d_model)      (seq, d_model)
                      .
                      .
                      .
    [ Input ] . . . . . . . .  [ K ]          @     [ Wk ]        =     [  K' ]
 (seq, d_model)       .    (seq, d_model)       (seq, d_model)      (seq, d_model)
                      .
                      .
                      .
                      . . . .  [ V ]          @     [ Wv ]        =     [  V' ]
                           (seq, d_model)       (seq, d_model)      (seq, d_model)
    ───────────────────────────────────────────────────────────────────────────────
    Step 2: Split into multiple heads

        Suppose:
            h = 8  heads
            d_model = 512 → each head has d_k = 64

        Then:
            Q → (batch, h, seq_len, d_k)  [same for K and V]

        Each head works with a different linear projection subspace.

        For example:
            Head 1 might learn syntactic dependencies (e.g., "it" ↔ "jazz")
            Head 2 might model sentence position (e.g., early vs. late emphasis)
            Head 3 might pick up semantic categories (e.g., verbs, adjectives)

                                                                                                   d_model
                                                                                             (d_k)  (d_k)  (d_k)
                      . . . .  [ Q ]          @     [ Wq ]        =     [  Q' ] . . . . . .  [Q1]    [Q2]   [Q3]
                      .    (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
    [ Input ] . . . . . . . .  [ K ]          @     [ Wk ]        =     [  K' ] . . . . . .  [K1]    [K2]   [K3]
 (seq, d_model)       .    (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      . . . .  [ V ]          @     [ Wv ]        =     [  V' ] . . . . . .  [V1]    [V2]   [V3]
                           (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                                                                                              .       .      .
                            Attention(Q, K, V) = softmax(Q @ Kᵀ/ sqrt(d_k) * V                .       .      .
                                    head i = Attention(QW(q,i), KW(k,i), VW(v,i))             .       .      .
                                                                                              .    d_model   .
                                                                                             (d_v)  (d_v)  (d_v)
                                                                                           [head1] [head2] [head3]
    ───────────────────────────────────────────────────────────────────────────────
    Step 3: Scaled Dot-Product Attention (per head)

        Each head runs:

            Attention(Q_i, K_i, V_i) = softmax(Q_i @ Kᵢᵀ / √d_k) @ V_i

        This is where ScaledDotProductAttentionBlock is applied.

        For example:

            Q₂ = embedding for "love" in Head 2
            K₂ = all keys in the sentence
            → score[i][j] = Q₂[i] · K₂[j] / √64

            Output["love"] in Head 2 =
                w₁·Value("I") + w₂·Value("love") + ... + w₇·Value("smoother")

        With 8 such heads, we learn 8 different ways "love" might relate to context.

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


        x = self.res_connections[0](x, lambda x: self.mha_block(q=x, k=x, v=x, mask=src_mask))

        - All tokens attend to each other:
            • "love" may attend to "jazz" and "smoother"
            • "it" may co-attend with "jazz" via coreference
            • "because" connects "love" and "smoother"
        - Heads model different relations (syntax, distance, emphasis)
        - Output: New vector for each token encoding global context

        Output ("love") ≈ f("love" + weights·["I", "jazz", "because", ...])
                                                                                                       d_model
                                                                                             (d_k)  (d_k)  (d_k)
                      . . . .  [ Q ]          @     [ Wq ]        =     [  Q' ] . . . . . .  [Q1]    [Q2]   [Q3]
                      .    (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
    [ Input ] . . . . . . . .  [ K ]          @     [ Wk ]        =     [  K' ] . . . . . .  [K1]    [K2]   [K3]
 (seq, d_model)       .    (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      . . . .  [ V ]          @     [ Wv ]        =     [  V' ] . . . . . .  [V1]    [V2]   [V3]
                           (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                                                                                              .       .      .
                            Attention(Q, K, V) = softmax(Q @ Kᵀ/ sqrt(d_k) * V                .       .      .
                                    head i = Attention(QW(q,i), KW(k,i), VW(v,i))             .       .      .
                                                                                              .    d_model   .
                                                                                             (d_v)  (d_v)  (d_v)
                                                                                           [head1] [head2] [head3] . . . . . .  [ H ]       x     [ Wo ]        =     [ MH-A ]
                                                                                                                           (seq, h * d_v)    (h * dvT d_model)    (seq, d_model)

                                                                                                               MultiHead(Q, K, V) = Concat(head1, ..., head h) x Wo


    ───────────────────────────────────────────────────────────────────────────────
    Stage 2: Residual Connection + LayerNorm

    ResidualConnectionLayer applies:
        output = x + Dropout(Sublayer(LayerNorm(x)))

    This stabilises training and preserves signal flow.
    It is applied around both MHA and FFN components.

    This block wraps any sublayer (e.g., attention or feedforward), standardises the input,
    adds dropout regularisation, and sums the original input to the result.

      Pre-norm design:
        Normalise before sublayer for better gradient flow and stable training.


      Applies normalisation over the last dimension of the input tensor
      (typically the embedding or feature dimension), followed by learnable
      affine transformation using `gamma` and `beta`.

      Example (after InputEmbeddingLayer + PositionalEncodingLayer):

        Sentence:        "I love jazz because it is smoother"
        Token Embeddings (dim=4):  [shown below before normalisation]

        Raw Input:
        ┌────────────┬────────────┬────────────┬────────────┐
        │   0.10     │   0.30     │  -0.20     │   0.50     │ ← "I"
        │   0.25     │  -0.10     │   0.40     │   0.05     │ ← "love"
        │  -0.30     │   0.60     │   0.20     │   0.10     │ ← "jazz"
        └────────────┴────────────┴────────────┴────────────┘

        Step-by-step per token (last dim is normalised):

        Token: "I"
        μ = mean([0.1, 0.3, -0.2, 0.5])   = 0.175
        σ² = var([...])                   ≈ 0.0675
        std = sqrt(σ² + eps)              ≈ 0.260

        Normalised:
        [(0.1 - 0.175) / 0.260, (0.3 - 0.175) / 0.260, ...]
        ≈ [-0.29, 0.48, -1.44, 1.25]

        Output after γ and β (default: γ=1, β=0):
        ≈ [-0.29, 0.48, -1.44, 1.25]

       Example (token: "jazz"):
        Let's say the self-attention or feedforward output vector for "jazz" is:
            sublayer(x) = [0.9, -1.1, 0.3, 0.5]
        The normalised input x was:
            norm(x)     = [0.5, -0.5, 0.0, 0.0]
        After dropout, we get:
            dropped     = [0.9,  0.0, 0.3, 0.5]   # some dims dropped
        Residual output:
            out         = input + dropped
                        = [0.5, -0.5, 0.0, 0.0] + [0.9, 0.0, 0.3, 0.5]
                        = [1.4, -0.5, 0.3, 0.5]

    ───────────────────────────────────────────────────────────────────────────────
    Stage 3: Position-wise feed forward network

    PositionWiseFeedForwardLayer applies:
        FF(x) = max(0, xW₁ + b₁)W₂ + b₂ (same for each token independently)

    It enhances the model’s ability to learn local transformations
    such as re-weighting attention features and introducing non-linearity.

    Applies residual connections with layer normalisation and dropout.

    This block wraps any sublayer (e.g., attention or feedforward), standardises the input,
    adds dropout regularisation, and sums the original input to the result.

       Pre-norm design:
        Normalise before sublayer for better gradient flow and stable training.

       Example (token: "jazz"):
        Let's say the self-attention or feedforward output vector for "jazz" is:
            sublayer(x) = [0.9, -1.1, 0.3, 0.5]
        The normalised input x was:
            norm(x)     = [0.5, -0.5, 0.0, 0.0]
        After dropout, we get:
            dropped     = [0.9,  0.0, 0.3, 0.5]   # some dims dropped
        Residual output:
            out         = input + dropped
                        = [0.5, -0.5, 0.0, 0.0] + [0.9, 0.0, 0.3, 0.5]
                        = [1.4, -0.5, 0.3, 0.5]

    This helps preserve input information and gradients over many layers.
    ───────────────────────────────────────────────────────────────────────────────
    Stage 4: Second residual connection + layer normalisation

    Another ResidualConnectionLayer is applied around the FFN.

    Final output is ready to be passed to the next encoder block (if any).

    ------------------------------------------------------------------------------
    The full encoder consists of multiple such EncoderBlock instances stacked.
    Each adds depth, representational power, and more abstraction.
    """
    def __init__(self, features: int, layers: ModuleList):
        super(EncoderLayer, self).__init__()
        self.layers = layers
        self.norm_layer = NormLayer(d_model=features)

    def forward(self, input: Tensor) -> Tensor:
        x = input
        for layer in self.layers:
            x = layer(x)
        x = self.norm_layer(x)
        return x
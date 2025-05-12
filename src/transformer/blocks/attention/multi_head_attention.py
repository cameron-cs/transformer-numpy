from src.nn import Module, Linear, Dropout
from src.tensor import Tensor
from src.transformer.blocks.attention.scaled_dot_product import ScaledDotProductAttentionBlock


class MultiHeadAttentionBlock(Module):
    """
    Multi-Head Attention block for the transformer architecture.

    This block computes attention scores using multiple attention heads in parallel.
    Each attention head computes a weighted sum of values based on queries and keys.
    The outputs of all attention heads are then concatenated and passed through a final linear layer.

    Args:
        h (int): The number of attention heads.
        d_model (int): The dimensionality of the model (input/output size).
        dropout (float, optional): Dropout probability for regularisation (default is 0.1).

    Attributes:
        h (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        attention (ScaledDotProductAttentionBlock): Scaled dot-product attention block.
        linear_wq (Linear): Linear layer for query transformation.
        linear_wk (Linear): Linear layer for key transformation.
        linear_wv (Linear): Linear layer for value transformation.
        linear_wo (Linear): Linear layer for output transformation after concatenating attention heads.

                                                                                                   d_model
                                                                                             (d_k)  (d_k)  (d_k)
                      . . . . [ Q ]          @     [ Wq ]        =     [  Q' ] . . . . . .   [Q1]    [Q2]   [Q3]
                      .    (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
    [ Input ] . . . . . . . . [ K ]          @     [ Wk ]        =     [  K' ] . . . . . .  [K1]    [K2]   [K3]
 (seq, d_model)       .    (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      .                                                                       .       .      .
                      . . . . [ V ]          @     [ Wv ]        =     [  V' ] . . . . . .  [V1]    [V2]   [V3]
                           (seq, d_model)       (seq, d_model)      (seq, d_model)            .       .      .
                                                                                              .       .      .
                            Attention(Q, K, V) = softmax(Q @ Kᵀ/ sqrt(d_k) * V                .       .      .
                                    head i = Attention(QW(q,i), KW(k,i), VW(v,i))             .       .      .
                                                                                              .    d_model   .
                                                                                             (d_v)  (d_v)  (d_v)
                                                                                            [head] [head]  [head] . . . . .  [ H ]       x     [ Wo ]        =     [ MH-A ]
                                                                                                                         (seq, h * d_v)    (h * dvT d_model)    (seq, d_model)

    """
    def __init__(self, h: int, d_model: int, dropout: float):
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.attention = ScaledDotProductAttentionBlock(dropout)
        self.linear_wq = Linear(d_model, d_model)  # Wq
        self.linear_wk = Linear(d_model, d_model)  # Wk
        self.linear_wv = Linear(d_model, d_model)  # Wv
        self.linear_wo = Linear(d_model, d_model)  # Wo

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
        """
        Perform the forward pass of the multi-head attention block.

        This method computes the attention scores for the given queries (q), keys (k),
        and values (v), splits the tensors by the number of heads, applies the attention mechanism,
        concatenates the outputs of the heads, and passes it through a final linear transformation.

        Args:
            q (Tensor): Query tensor of shape [batch_size, length, d_model].
            k (Tensor): Key tensor of shape [batch_size, length, d_model].
            v (Tensor): Value tensor of shape [batch_size, length, d_model].
            mask (Tensor, optional): Mask tensor to prevent attention to certain positions (default is None).

        Returns:
            Tensor: The output of multi-head attention with shape [batch_size, length, d_model].
        """
        # 1. dot product with weight matrices
        # Q'(seq,d_model) = Q(seq,d_model) x Wq(d_model,d_model)
        q: Tensor = self.linear_wq(q)
        # K'(seq,d_model) = K(seq,d_model) x Wk(d_model,d_model)
        k: Tensor = self.linear_wk(k)
        # V'(seq,d_model) = V(seq,d_model) x Wv(d_model,d_model)
        v: Tensor = self.linear_wv(v)

        # 2. split tensor by number of heads
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        # we split embeddings not sentences
        q: Tensor = self.split(q)
        k: Tensor = self.split(k)
        v: Tensor = self.split(v)

        # 3. do scale dot product to compute similarity
        out, _ = self.attention(q, k, v, mask)

        # 4. concat and pass to linear layer
        out: Tensor = self.concat(out)
        out: Tensor = self.linear_wo(out)

        return out

    def split(self, tensor: Tensor) -> Tensor:
        """
        Split the input tensor into multiple heads.

        This function reshapes the input tensor so that it can be processed by multiple attention heads
        in parallel, with each head receiving a fraction of the model's dimensionality.

        Args:
            tensor (Tensor): The input tensor with shape [batch_size, length, d_model].

        Returns:
            Tensor: The reshaped tensor with shape [batch_size, head, length, d_k],
                    where d_tensor = d_model / h is the dimensionality of each attention head.
        """
        tensor = tensor.view(tensor.shape()[0], tensor.shape()[1], self.h, self.d_k).transpose(1, 2)
        return tensor

    def concat(self, tensor: Tensor) -> Tensor:
        """
        Concatenate the outputs of all attention heads.

        This function reverses the splitting operation by combining the outputs of all attention heads
        back into a single tensor.

        Args:
            tensor (Tensor): The tensor representing the outputs of the attention heads,
                             with shape [batch_size, head, seq_len, dk].

        Returns:
            Tensor: The concatenated tensor with shape [batch_size, seq_len, d_model].
        """
        tensor: Tensor = tensor.transpose(1, 2).contiguous().view(tensor.shape()[0], -1, self.h * self.d_k)
        return tensor

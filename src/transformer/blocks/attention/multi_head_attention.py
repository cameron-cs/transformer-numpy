from src.nn import Module, Linear, Dropout
from src.tensor import Tensor
from src.transformer.blocks.attention.scaled_dot_product import ScaledDotProductAttentionBlock


class MultiHeadAttentionBlock(Module):

    def __init__(self, h, d_model, dropout_p=0.1):
        self.h = h
        self.d_model = d_model
        self.attention = ScaledDotProductAttentionBlock()
        self.linear_wq = Linear(d_model, d_model)
        self.linear_wk = Linear(d_model, d_model)
        self.linear_wv = Linear(d_model, d_model)
        self.linear_wo = Linear(d_model, d_model)

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', mask=None) -> 'Tensor':
        # 1. dot product with weight matrices
        q: Tensor = self.linear_wq(q)
        k: Tensor = self.linear_wk(k)
        v: Tensor = self.linear_wv(v)

        # 2. split tensor by number of heads
        q: Tensor = self.split(q)
        k: Tensor = self.split(k)
        v: Tensor = self.split(v)

        # 3. do scale dot product to compute similarity
        out, _ = self.attention(q, k, v, mask)

        # 4. concat and pass to linear layer
        out: Tensor = self.concat(out)
        out: Tensor = self.linear_wo(out)

        return out

    def split(self, tensor: 'Tensor') -> 'Tensor':
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        print("tensor.size()", tensor.size())
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.h
        tensor = tensor.view(batch_size, length, self.h, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor: 'Tensor'):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

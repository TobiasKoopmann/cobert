import torch

from model.attention.multi_head import MultiHeadedAttention
from model.utils.sublayer import SublayerConnection
from model.utils.feed_forward import PositionwiseFeedForward


class TransformerBlock(torch.nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden_size, n_heads, intermediate_size, p_dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=n_heads, d_model=hidden_size, dropout=p_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden_size, d_ff=intermediate_size, dropout=p_dropout)
        self.input_sublayer = SublayerConnection(size=hidden_size, dropout=p_dropout)
        self.output_sublayer = SublayerConnection(size=hidden_size, dropout=p_dropout)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, x, mask) -> torch.Tensor:
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class NovaTransformerBlock(torch.nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden_size, n_heads, intermediate_size, p_dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=n_heads, d_model=hidden_size, dropout=p_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden_size, d_ff=intermediate_size, dropout=p_dropout)
        self.input_sublayer = SublayerConnection(size=hidden_size, dropout=p_dropout)
        self.output_sublayer = SublayerConnection(size=hidden_size, dropout=p_dropout)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, x, meta, mask) -> torch.Tensor:
        """
        :param x: batch of inputs
        :param meta: batch of meta-information
        :param mask:
        :return:
        """
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, meta, meta, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

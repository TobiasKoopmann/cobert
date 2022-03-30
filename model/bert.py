from torch import nn as nn

from model.embedding import *
from model.attention.transformer import TransformerBlock, NovaTransformerBlock
from model.utils import fix_random_seed_as


class Bert4RecOG(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_len: int = 200,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 hidden_size: int = 256,
                 p_dropout: float = 0.1,
                 seed: int = 123):
        super().__init__()

        fix_random_seed_as(seed)
        # self.init_weights()

        self.max_len = max_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout

        self.embedding = BertEmbeddingAE(vocab_size=self.vocab_size,
                                         embed_size=self.hidden_size,
                                         max_len=self.max_len,
                                         dropout=self.p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size=self.hidden_size,
                              n_heads=self.n_heads,
                              intermediate_size=self.hidden_size * 4,
                              p_dropout=self.p_dropout)
             for _ in range(n_layers)]
        )

        self.projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.projection_activation = nn.GELU()
        self.projection_norm = nn.LayerNorm(self.hidden_size)
        self.out_bias = nn.Parameter(torch.zeros(self.vocab_size))

    def forward(self, batch):
        x = self.embedding(batch["author_ids"], batch["position_ids"])

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, batch["attention_mask"])

        x = self.projection(x)
        x = self.projection_activation(x)
        x = self.projection_norm(x)

        # weight tying...
        x = x @ self.embedding.token.weight.T + self.out_bias

        return x


class Bert4RecAE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_len: int = 200,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 hidden_size: int = 256,
                 p_dropout: float = 0.1,
                 seed: int = 123):
        super().__init__()

        fix_random_seed_as(seed)
        # self.init_weights()

        self.max_len = max_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout

        self.embedding = BertEmbeddingAE(vocab_size=self.vocab_size,
                                         embed_size=self.hidden_size,
                                         max_len=self.max_len,
                                         dropout=self.p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size=self.hidden_size,
                              n_heads=self.n_heads,
                              intermediate_size=self.hidden_size * 4,
                              p_dropout=self.p_dropout)
             for _ in range(n_layers)]
        )

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, batch):
        x = self.embedding(batch["author_ids"], batch["position_ids"])

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, batch["attention_mask"])

        return self.out(x)


class Bert4RecAEW(Bert4RecAE):
    def __init__(self,
                 vocab_size: int,
                 max_len: int = 200,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 hidden_size: int = 256,
                 p_dropout: float = 0.1,
                 seed: int = 123):
        super().__init__(vocab_size, max_len, n_layers, n_heads, hidden_size, p_dropout, seed)

        self.embedding = BertEmbeddingAEW(vocab_size=self.vocab_size,
                                          embed_size=self.hidden_size,
                                          max_len=self.max_len,
                                          dropout=self.p_dropout)


class Bert4RecAEPE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 n_papers: int,
                 max_len: int = 200,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 hidden_size: int = 256,
                 p_dropout: float = 0.1,
                 seed: int = 123):
        super().__init__()

        fix_random_seed_as(seed)

        self.max_len = max_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.n_papers = n_papers
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout

        self.embedding = BertEmbeddingAEPE(vocab_size=self.vocab_size,
                                           n_papers=self.n_papers,
                                           embed_size=self.hidden_size,
                                           max_len=self.max_len,
                                           dropout=self.p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size=self.hidden_size,
                              n_heads=self.n_heads,
                              intermediate_size=self.hidden_size * 4,
                              p_dropout=self.p_dropout)
             for _ in range(n_layers)]
        )
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, batch):
        x = self.embedding(batch["author_ids"], batch["position_ids"], batch["paper_ids"])

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, batch["attention_mask"])

        return self.out(x)


class Bert4RecAEPEW(Bert4RecAEPE):
    def __init__(self,
                 vocab_size: int,
                 n_papers,
                 max_len: int = 200,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 hidden_size: int = 256,
                 p_dropout: float = 0.1,
                 seed: int = 123):
        super().__init__(vocab_size, n_papers, max_len, n_layers, n_heads, hidden_size, p_dropout, seed)

        self.embedding = BertEmbeddingAEPEW(vocab_size=self.vocab_size,
                                            n_papers=self.n_papers,
                                            embed_size=self.hidden_size,
                                            max_len=self.max_len,
                                            dropout=self.p_dropout)


class Bert4RecAEPESeq(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 n_papers: int,
                 max_len: int = 200,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 hidden_size: int = 256,
                 p_dropout: float = 0.1,
                 seed: int = 123):
        super().__init__()

        fix_random_seed_as(seed)

        self.max_len = max_len
        self.n_layers = n_layers
        self.n_papers = n_papers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout

        self.embedding = BertEmbeddingAEPESeq(vocab_size=self.vocab_size,
                                              n_papers=self.n_papers,
                                              embed_size=self.hidden_size,
                                              max_len=self.max_len,
                                              dropout=self.p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size=self.hidden_size,
                              n_heads=self.n_heads,
                              intermediate_size=self.hidden_size * 4,
                              p_dropout=self.p_dropout)
             for _ in range(n_layers)]
        )
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, batch):
        x = self.embedding(batch["author_ids"], batch["position_ids"], batch["segment_ids"], batch["paper_ids"])

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, batch["attention_mask"])
        return self.out(x)


class Bert4RecNova(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 n_papers: int,
                 max_len: int = 200,
                 n_layers: int = 2,
                 n_heads: int = 4,
                 hidden_size: int = 256,
                 p_dropout: float = 0.1,
                 seed: int = 123):
        super().__init__()

        fix_random_seed_as(seed)

        self.max_len = max_len
        self.n_layers = n_layers
        self.n_papers = n_papers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout

        self.embedding = BertEmbeddingNova(vocab_size=self.vocab_size,
                                           n_papers=self.n_papers,
                                           embed_size=self.hidden_size,
                                           max_len=self.max_len,
                                           dropout=self.p_dropout)

        self.transformer_blocks = nn.ModuleList(
            [NovaTransformerBlock(hidden_size=self.hidden_size,
                                  n_heads=self.n_heads,
                                  intermediate_size=self.hidden_size * 4,
                                  p_dropout=self.p_dropout)
             for _ in range(n_layers)]
        )
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, batch):
        x, meta = self.embedding(batch["author_ids"], batch["position_ids"], batch["paper_ids"])

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, meta, batch["attention_mask"])

        return self.out(x)

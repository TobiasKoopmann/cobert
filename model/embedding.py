import torch


class BertEmbeddingAE(torch.nn.Module):
    """ only author embeddings """
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):  # n_papers
        super().__init__()
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = torch.nn.Embedding(max_len, embed_size, padding_idx=0)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, position_ids):
        x = self.token(sequence) + self.position(position_ids)

        return self.dropout(x)


class BertEmbeddingAEPE(torch.nn.Module):
    """ author + paper embeddings """
    def __init__(self, vocab_size, n_papers, embed_size, max_len, dropout=0.1):  # n_papers
        super().__init__()
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = torch.nn.Embedding(max_len, embed_size, padding_idx=0)
        self.paper = torch.nn.Embedding(n_papers, embed_size, padding_idx=0)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, position_ids, paper_ids):
        x = self.token(sequence) + self.position(position_ids) + self.paper(paper_ids)

        return self.dropout(x)


class BertEmbeddingAEW(torch.nn.Module):
    """ weighted author embeddings """
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):  # n_papers
        super().__init__()
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = torch.nn.Embedding(max_len, embed_size, padding_idx=0)

        self.dropout = torch.nn.Dropout(p=dropout)

        self.embedding_weights = torch.nn.Parameter(torch.empty((2, embed_size)))
        self.embedding_weights.data.normal_(mean=0.0, std=0.02)
        self.embedding_bias = torch.nn.Parameter(torch.zeros(embed_size))

    def forward(self, sequence, position_ids):
        token_emb = self.token(sequence).unsqueeze(2)
        pos_emb = self.position(position_ids).unsqueeze(2)

        embeddings_concat = torch.cat((token_emb, pos_emb), dim=2)
        x = torch.einsum("bsed,ed->bsd", embeddings_concat, self.embedding_weights)
        x = x[:, ] + self.embedding_bias

        return self.dropout(x)


class BertEmbeddingAEPEW(torch.nn.Module):
    """ weighted author + paper embeddings """
    def __init__(self, vocab_size, n_papers, embed_size, max_len, dropout=0.1):  # n_papers
        super().__init__()
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = torch.nn.Embedding(max_len, embed_size, padding_idx=0)
        self.paper = torch.nn.Embedding(n_papers, embed_size, padding_idx=0)

        self.dropout = torch.nn.Dropout(p=dropout)

        # self.embedding_weights = torch.nn.Parameter(torch.ones(3))
        self.embedding_weights = torch.nn.Parameter(torch.empty((3, embed_size)))
        self.embedding_weights.data.normal_(mean=0.0, std=0.02)
        self.embedding_bias = torch.nn.Parameter(torch.zeros(embed_size))

    def forward(self, sequence, position_ids, paper_ids):
        # token_emb = self.token(sequence).unsqueeze(2) * self.embedding_weights[0]
        # paper_emb = self.paper(paper_ids).unsqueeze(2) * self.embedding_weights[1]
        # pos_emb = self.position(position_ids).unsqueeze(2) * self.embedding_weights[2]
        token_emb = self.token(sequence).unsqueeze(2)
        paper_emb = self.paper(paper_ids).unsqueeze(2)
        pos_emb = self.position(position_ids).unsqueeze(2)
        embeddings_concat = torch.cat((token_emb, paper_emb, pos_emb), dim=2)
        # x = torch.sum(embeddings_concat, 2)
        x = torch.einsum("bsed,ed->bsd", embeddings_concat, self.embedding_weights)
        x = x[:, ] + self.embedding_bias

        return self.dropout(x)


class BertEmbeddingAEPESeq(torch.nn.Module):
    """ sequential author + paper embeddings """
    def __init__(self, vocab_size, n_papers, embed_size, max_len, dropout=0.1):  # n_papers
        super().__init__()
        self.paper = torch.nn.Embedding(n_papers, embed_size)
        self.token = torch.nn.Embedding(vocab_size, embed_size)
        self.segment = torch.nn.Embedding(3, embed_size)
        self.position = torch.nn.Embedding(max_len, embed_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, position_ids, segment_ids, paper_ids):
        seq_emb = self.token(sequence)
        seq_emb[sequence == 0, :] = 0
        paper_emb = self.paper(paper_ids)
        paper_emb[paper_ids == 0, :] = 0
        paper_emb[:, :seq_emb.shape[1], :] += seq_emb
        paper_emb += self.position(position_ids)
        paper_emb += self.segment(segment_ids)

        return self.dropout(paper_emb)


class BertEmbeddingNova(torch.nn.Module):
    """ NovaBert Embeddings """
    def __init__(self, vocab_size, n_papers, embed_size, max_len, dropout=0.1):  # n_papers
        super().__init__()
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = torch.nn.Embedding(max_len, embed_size, padding_idx=0)
        self.paper = torch.nn.Embedding(n_papers, embed_size, padding_idx=0)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, position_ids, paper_ids):
        x = self.token(sequence)
        y = self.position(position_ids) + self.paper(paper_ids)
        return self.dropout(x), self.dropout(y)

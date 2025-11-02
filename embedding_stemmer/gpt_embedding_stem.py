"""embedding_stemmer/gpt_embedding_stem.py

Implementation of the GPT-style token and positional embedding stem.
"""

import torch
from torch import nn


class GPTEmbeddingStem(nn.Module):
    """Token and positional embedding stem for GPT-style language models."""

    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        return token_embeddings + position_embeddings


__all__ = ["GPTEmbeddingStem"]

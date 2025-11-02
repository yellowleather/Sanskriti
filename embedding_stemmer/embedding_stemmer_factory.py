"""embedding_stemmer/embedding_stemmer_factory.py

Factory functions for creating embedding stems.
"""

from typing import Literal

from torch import nn

from embedding_stemmer.gpt_embedding_stem import GPTEmbeddingStem


def get_embedding_stem(
    stem_type: Literal["gpt"] = "gpt",
    *,
    vocab_size: int,
    embedding_dim: int,
    context_length: int,
) -> nn.Module:
    """Return an embedding stem implementation by name."""

    if stem_type == "gpt":
        return GPTEmbeddingStem(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
        )

    raise ValueError(
        f"Unknown embedding stem '{stem_type}'. Supported stems: gpt"
    )


__all__ = ["get_embedding_stem"]

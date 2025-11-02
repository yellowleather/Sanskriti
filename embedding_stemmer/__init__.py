"""embedding_stemmer package

Provides factory helpers for constructing embedding stems used in the language model pipeline.
"""

from embedding_stemmer.embedding_stemmer_factory import get_embedding_stem
from embedding_stemmer.gpt_embedding_stem import GPTEmbeddingStem

__all__ = ["GPTEmbeddingStem", "get_embedding_stem"]

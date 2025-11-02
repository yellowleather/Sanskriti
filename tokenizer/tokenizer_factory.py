"""tokenizer/tokenizer_factory.py

Small factory for returning a tokenizer object for this project.

At present this returns a BPE encoding from the `tiktoken` library (GPT-2 BPE by
default). The returned object is the encoding returned by `tiktoken.get_encoding()`
and exposes `.encode()` / `.decode()` and related helpers.

BPE (Byte Pair Encoding) is a tokenization algorithm that builds a vocabulary by
iteratively merging the most frequently occurring pairs of tokens in a corpus.
It starts with individual characters/bytes and creates progressively larger tokens
for common subwords and words. This allows the tokenizer to handle any text while
efficiently representing common patterns with single tokens and rare words as
sequences of subword pieces.

This module intentionally keeps the surface small so callers can swap implementations
later without changing call sites.
"""
from typing import Any


def get_tokenizer(encoding_name: str = "gpt2") -> Any:
    """Return a BPE tokenizer/encoding object from the tiktoken library.

    Parameters
    - encoding_name: name of the tiktoken encoding to return (default: "gpt2").

    Returns
    - encoding object from tiktoken (has .encode/.decode methods)

    Raises
    - ImportError: if tiktoken is not installed
    - RuntimeError: if tiktoken fails to provide the requested encoding
    """
    try:
        import tiktoken  # type: ignore
    except Exception as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "tiktoken is required for tokenizer_factory.get_tokenizer(). "
            "Install it with: pip install tiktoken"
        ) from e

    # Prefer get_encoding (returns a BPE encoding for 'gpt2').
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        # Fall back to encoding_for_model which some users prefer.
        try:
            return tiktoken.encoding_for_model(encoding_name)
        except Exception as e:  # pragma: no cover - runtime failure
            raise RuntimeError(f"failed to obtain tokenizer encoding '{encoding_name}': {e}") from e


__all__ = ["get_tokenizer"]


if __name__ == "__main__":
    # Quick smoke run (won't import tiktoken at compile-time) — prints a short sample.
    try:
        enc = get_tokenizer()
        sample = "The quick brown fox jumps over the lazy dog"
        toks = enc.encode(sample)
        print(f"Encoding '{enc.name}' produced {len(toks)} tokens; sample: {toks[:20]}")
    except Exception as e:
        print("Failed to construct tokenizer:", e)

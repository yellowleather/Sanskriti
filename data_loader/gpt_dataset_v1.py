"""data_loader/gpt_dataset_v1.py

GPT-style dataset implementation that creates input-target pairs with sliding window.
"""

import torch
from torch.utils.data import Dataset
from typing import Any


class GPTDatasetV1(Dataset):
    """Dataset for GPT-style language modeling with sliding window approach.

    This dataset tokenizes text and creates overlapping sequences of input-target
    pairs where targets are shifted by one token (next-token prediction).

    Uses a sliding window with configurable stride to create training examples from
    a continuous text corpus.
    """

    def __init__(self, txt: str, tokenizer: Any, max_length: int, stride: int):
        """Initialize GPTDatasetV1.

        Parameters:
        - txt: Raw text to tokenize and create dataset from
        - tokenizer: Tokenizer with encode() method (e.g., tiktoken encoding)
        - max_length: Maximum sequence length for each training example
        - stride: Step size for the sliding window (controls overlap)

        Raises:
        - AssertionError: If tokenized text is shorter than max_length + 1
        """
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # Note: tiktoken's encode doesn't have allowed_special parameter by default
        # If you need special tokens, this can be added to tokenizer configuration
        token_ids = tokenizer.encode(txt)

        if len(token_ids) <= max_length:
            raise ValueError(
                f"Number of tokenized inputs ({len(token_ids)}) must be greater than "
                f"max_length ({max_length})"
            )

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Return the number of training examples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input-target pair at the given index.

        Parameters:
        - idx: Index of the training example

        Returns:
        - tuple: (input_tensor, target_tensor) where target is shifted by 1
        """
        return self.input_ids[idx], self.target_ids[idx]


__all__ = ["GPTDatasetV1"]

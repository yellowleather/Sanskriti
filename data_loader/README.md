# Data Loader Module

This module provides PyTorch datasets and data loaders for LLM training, implementing a sliding window approach for creating training examples from continuous text.

## Key Concepts

### Parameters Explained

#### **max_length** (Sequence Length)
- **What**: The length of each training sequence in tokens
- **Purpose**: Defines how many tokens the model processes at once (the "context window")
- **Example**: If `max_length=256`, each training example contains 256 tokens
- **Training Impact**:
  - Longer sequences = more context but higher memory/computation requirements
  - Typical values: 128, 256, 512, 1024

#### **stride** (Sliding Window Step)
- **What**: How many tokens to skip between consecutive training samples
- **Purpose**: Controls overlap between training examples
- **Example with max_length=256, stride=128**:
  ```
  Sample 1: tokens[0:256]     (tokens 0-255)
  Sample 2: tokens[128:384]   (tokens 128-383) ← 128 tokens overlap!
  Sample 3: tokens[256:512]   (tokens 256-511) ← another 128 overlap
  ```
- **Why Overlap?**:
  - Smaller stride = more training samples, better data coverage
  - `stride=128` with `max_length=256` = 50% overlap
  - `stride=256` = no overlap (each token seen once)
  - Smaller stride = slower training but more thorough learning

#### **batch_size** (Training Batch)
- **What**: How many sequences to process together in one forward/backward pass
- **Purpose**: Training efficiency and gradient estimation
- **Example**: `batch_size=4` means process 4 sequences simultaneously
- **Shape**: With batch_size=4 and max_length=256 → `torch.Size([4, 256])`
- **Training Impact**:
  - Larger batches = faster training, more stable gradients, higher GPU memory
  - Smaller batches = less memory, noisier gradients, better generalization
  - Typical values: 4, 8, 16, 32

## Visual Example

```python
Text: "The quick brown fox jumps over the lazy dog and runs fast"
Tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

max_length = 4
stride = 2

# Dataset creates these samples:
Sample 1: input=[1,2,3,4]    target=[2,3,4,5]
Sample 2: input=[3,4,5,6]    target=[4,5,6,7]   ← 2 tokens overlap
Sample 3: input=[5,6,7,8]    target=[6,7,8,9]   ← 2 tokens overlap
...

# DataLoader groups them:
Batch 1: [[1,2,3,4], [3,4,5,6]]  (batch_size=2)
Batch 2: [[5,6,7,8], [7,8,9,10]]
```

## How It Works

The `GPTDatasetV1` class implements next-token prediction:

1. **Tokenizes** the entire training text
2. **Slides a window** of size `max_length` across tokens with step size `stride`
3. For each window:
   - **Input**: tokens[i : i+max_length]
   - **Target**: tokens[i+1 : i+max_length+1] (shifted by 1 for next-token prediction)
4. **Batches** the data for efficient training

## Common Configurations

```bash
# Lots of overlap, small batches (good for small datasets)
python main.py --max-length 256 --stride 64 --batch-size 2

# No overlap, large batches (efficient training)
python main.py --max-length 256 --stride 256 --batch-size 8

# Medium overlap (balanced - DEFAULT)
python main.py --max-length 256 --stride 128 --batch-size 4
```

## Configuration Guidelines

| Parameter | Small Dataset | Medium Dataset | Large Dataset |
|-----------|--------------|----------------|---------------|
| **max_length** | 128-256 | 256-512 | 512-1024 |
| **stride** | max_length/4 | max_length/2 | max_length |
| **batch_size** | 2-4 | 8-16 | 32-64 |

## Module Structure

```
data_loader/
├── __init__.py                    # Exports create_dataset, create_dataloader
├── data_loader_factory.py         # Factory functions
├── gpt_dataset_v1.py              # GPTDatasetV1 implementation
└── README.md                      # This file
```

## Usage

```python
from data_loader import create_dataset, create_dataloader

# Create dataset
dataset = create_dataset(
    dataset_type="gpt_v1",
    txt=raw_text,
    tokenizer=tokenizer,
    max_length=256,
    stride=128,
)

# Create dataloader
dataloader = create_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True,
)

# Iterate
for inputs, targets in dataloader:
    # inputs.shape = [batch_size, max_length]
    # targets.shape = [batch_size, max_length]
    # targets are inputs shifted by 1 token
    ...
```

## Performance Considerations

- **Memory**: `batch_size × max_length × model_size` determines GPU memory usage
- **Training Speed**: Larger `batch_size` improves throughput but requires more memory
- **Data Coverage**: Smaller `stride` means each token is seen in more contexts
- **Overfitting**: Higher overlap (smaller `stride`) can increase overfitting on small datasets

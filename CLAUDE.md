# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements toy transformer models using bilinear layers and quadratic attention (squared attention patterns) to study the mathematical structure of small transformer architectures. The goal is to replicate interpretability techniques for 1-2 layer transformer and attention-only architectures using tensor network decomposition methods.

## Key Components

### Core Architecture (`utils.py`)

The main model implementation uses:
- **QuadraticAttention**: Attention mechanism using quadratic scoring (`(q·k)²`) instead of softmax
- **BilinearLayer**: Element-wise product of two linear projections (similar to SwiGLU without gating)
- **ToyTransformer**: Flexible transformer supporting `attention_only_1L`, `attention_only_2L`, `transformer_1L`, `transformer_2L` configurations
- **Muon Optimizer**: Memory-efficient optimizer with ~1.5x better sample efficiency than Adam
- **RoPE (Rotary Positional Embeddings)**: Used instead of learned positional embeddings

### Tokenization (`tokenization/tokenization.py`)

Uses a 10k vocabulary tokenizer from TinyStories (from Noa Nabeshima's repo). Key functions:
- `enc()`: Encode text to token IDs with attention masks
- `dec()`: Decode token IDs to text
- `tokenizer` object provides standard `.encode()` and `.decode()` methods
- Vocab size: 10,000 tokens

### Training Infrastructure

**StreamingTextDataset**: Streams data from HuggingFace datasets (FineWeb, TinyStories, etc.) with:
- Deterministic train/validation splitting using hash-based filtering
- On-the-fly tokenization with buffer management
- Handles padding tokens and creates proper attention masks

**Trainer**: Training loop with:
- Mixed precision training (AMP)
- Gradient clipping
- Cosine learning rate schedule with warmup
- Evaluation on validation set

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source tn_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training Models

Train models using the Python script with command-line options:
```bash
# Train on SimpleStories (default, 4096 vocab)
python toy_transformer_trainer.py --dataset simplestories --model_type attention_only_1L

# Train on TinyStories (10k vocab)
python toy_transformer_trainer.py --dataset tinystories --model_type transformer_1L

# Train on FineWeb (GPT-2 tokenizer, 50257 vocab)
python toy_transformer_trainer.py --dataset fineweb --model_type transformer_2L

# Additional options
python toy_transformer_trainer.py --dataset simplestories \
    --model_type attention_only_1L \
    --batch_size 32 \
    --learning_rate 3e-3 \
    --max_iters 10000

# Or use Jupyter notebooks
jupyter notebook toy_trainer.ipynb
```

Available datasets:
- `simplestories`: SimpleStories dataset with 4096 vocab (best for experiments)
- `tinystories`: TinyStories dataset with 10k vocab
- `fineweb`: FineWeb dataset with GPT-2 tokenizer (50257 vocab)

### Model Configurations

Models are configured via `ModelConfig` dataclass:
- `model_type`: `'attention_only_1L'`, `'attention_only_2L'`, `'transformer_1L'`, `'transformer_2L'`
- `vocab_size`: Tokenizer vocabulary size (10000 for TinyStories, 4096 for SimpleStories)
- `d_model`: Model dimension (e.g., 512)
- `n_head`: Number of attention heads
- `n_ctx`: Context window length
- `dropout`: Dropout rate

### Loading Trained Models

```python
import torch
import json
from utils import ToyTransformer, ModelConfig

# Load config (format: {dataset}_{model_type}_config.json)
with open('simplestories_attention_only_1L_config.json', 'r') as f:
    config_dict = json.load(f)
config = ModelConfig(**config_dict)

# Load model (format: toy_transformer_{dataset}_{model_type}.pt)
model = ToyTransformer(config)
state_dict = torch.load('toy_transformer_simplestories_attention_only_1L.pt')
model.load_state_dict(state_dict)
```

## Mathematical Framework

The repository explores tensor network decompositions of attention mechanisms. Key concepts in `mathematical_framework.ipynb`:

### Circuit Analysis

For attention-only models, the logit output can be decomposed as:
- **QK Circuit**: `E @ Q.T @ K @ E.T` (attention patterns, symmetric matrix)
- **OV Circuit**: `U @ V.T @ O.T @ E.T` (value transformations)
- **Full Circuit**: Contracts through query, key, value, and output matrices

Uses `quimb.tensor` for tensor network representations and contractions.

### Visualization

`visualize_prediction_error()` in `utils.py` generates HTML visualizations showing per-token cross-entropy loss with white-to-red gradient (saturating at 7.0).

## Architecture Details

### Quadratic Attention Mechanism

Unlike standard softmax attention, this uses:
1. Compute scores: `(Q @ K.T) / d_head`
2. Square the scores: `scores²`
3. Apply causal mask
4. Aggregate values: `pattern @ V`

This creates polynomial (degree-2) attention patterns amenable to tensor network analysis.

### Weight Initialization

- Linear layers: Normal initialization (std=0.02)
- Output projections: Zero-initialized for stability (muP-like)
- Embeddings: Normal initialization (std=0.02)
- Weight tying: Embedding and output head share weights

## Notebooks

- `toy_trainer.ipynb`: Training loop with SimpleStories dataset, visualization of prediction errors
- `mathematical_framework.ipynb`: Tensor network analysis, circuit decomposition (QK/OV circuits), quimb tensor network experiments
- `deep_bilinear_training_dynamics.ipynb`: Studies of training dynamics

## Important Notes

- The codebase uses RMSNorm but it's often commented out in favor of no normalization
- Models use causal masking for autoregressive generation
- Attention masks are properly handled during loss calculation (only computing loss on non-padded tokens)
- The tokenizer vocab size must match the model's `vocab_size` parameter

## Data Sources

- TinyStories: 10k vocab tokenizer, simple children's stories
- SimpleStories: 4096 vocab, used in recent training runs
- FineWeb: Larger web corpus for scaling experiments

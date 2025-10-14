#!/usr/bin/env python3
"""
Load a trained toy transformer model and SimpleStories dataset.
This script demonstrates how to properly load a saved model and access the dataset.
"""

import sys
import os
# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from utils import ToyTransformer, ModelConfig, StreamingTextDataset

# ============================================================================
# Configuration
# ============================================================================

# Get the root directory (parent of scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model and config paths (relative to root)
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'toy_transformer_simplestories_transformer_3L_10000batches.pt')
CONFIG_PATH = os.path.join(ROOT_DIR, 'configs', 'simplestories_transformer_3L_10000batches_config.json')

# Dataset configuration (matching trainer)
DATASET_CONFIG = {
    'dataset_name': 'SimpleStories/SimpleStories',
    'subset': None,
    'tokenizer_name': 'SimpleStories/SimpleStories-1.25M',
    'vocab_size': 4096,
}

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Load Model
# ============================================================================

def load_model():
    """Load the trained model from config and weights."""
    print("=" * 60)
    print("Loading Model")
    print("=" * 60)

    # Load config
    print(f"Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config_dict = json.load(f)

    # Create ModelConfig (only use the fields that ModelConfig expects)
    model_config = ModelConfig(
        vocab_size=config_dict['vocab_size'],
        d_model=config_dict['d_model'],
        n_ctx=config_dict['n_ctx'],
        n_head=config_dict['n_head'],
        dropout=config_dict['dropout'],
        model_type=config_dict['model_type']
    )

    print(f"Model type: {model_config.model_type}")
    print(f"Vocab size: {model_config.vocab_size}")
    print(f"d_model: {model_config.d_model}")
    print(f"n_ctx: {model_config.n_ctx}")
    print(f"n_head: {model_config.n_head}")

    # Create model
    model = ToyTransformer(model_config)

    # Load weights
    print(f"\nLoading weights from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # Move to device and set to eval mode
    model.to(DEVICE)
    model.eval()

    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Device: {DEVICE}")

    # Print training info if available
    if 'num_batches' in config_dict:
        print(f"\nTraining Info:")
        print(f"  Batches trained: {config_dict['num_batches']}")
        print(f"  Final train loss: {config_dict.get('final_train_loss', 'N/A'):.4f}")
        print(f"  Final val loss: {config_dict.get('final_val_loss', 'N/A'):.4f}")

    print("\n✓ Model loaded successfully!\n")
    return model, model_config

# ============================================================================
# Load Dataset
# ============================================================================

def load_dataset(model_config):
    """Initialize the SimpleStories dataset."""
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    print(f"Dataset: {DATASET_CONFIG['dataset_name']}")
    print(f"Tokenizer: {DATASET_CONFIG['tokenizer_name']}")
    print(f"Vocab size: {DATASET_CONFIG['vocab_size']}")
    print(f"Sequence length: {model_config.n_ctx}")

    # Create train dataset
    print("\nInitializing train dataset...")
    train_dataset = StreamingTextDataset(
        dataset_name=DATASET_CONFIG['dataset_name'],
        subset=DATASET_CONFIG['subset'],
        split='train',
        tokenizer_name=DATASET_CONFIG['tokenizer_name'],
        seq_length=model_config.n_ctx,
        validation_ratio=0.001
    )

    # Create validation dataset
    print("Initializing validation dataset...")
    val_dataset = StreamingTextDataset(
        dataset_name=DATASET_CONFIG['dataset_name'],
        subset=DATASET_CONFIG['subset'],
        split='validation',
        tokenizer_name=DATASET_CONFIG['tokenizer_name'],
        seq_length=model_config.n_ctx,
        validation_ratio=0.001
    )

    print("\n✓ Datasets loaded successfully!\n")
    return train_dataset, val_dataset

# ============================================================================
# Test Function
# ============================================================================

def test_model_and_data(model, train_dataset, val_dataset, model_config):
    """Test that the model and data work together."""
    print("=" * 60)
    print("Testing Model and Data")
    print("=" * 60)

    # Get a batch from train dataset
    print("Getting batch from train dataset...")
    x_train, y_train = train_dataset.get_batch(batch_size=4, device=DEVICE)
    print(f"Train batch shape: x={x_train.shape}, y={y_train.shape}")

    # Run model on train batch
    print("\nRunning model on train batch...")
    with torch.no_grad():
        attention_mask = torch.ones_like(x_train)
        logits, loss = model(x_train, attention_mask)

    print(f"Logits shape: {logits.shape}")
    print(f"Train loss: {loss.item():.4f}")

    # Get a batch from validation dataset
    print("\nGetting batch from validation dataset...")
    x_val, y_val = val_dataset.get_batch(batch_size=4, device=DEVICE)
    print(f"Val batch shape: x={x_val.shape}, y={y_val.shape}")

    # Run model on validation batch
    print("\nRunning model on validation batch...")
    with torch.no_grad():
        attention_mask = torch.ones_like(x_val)
        logits, loss = model(x_val, attention_mask)

    print(f"Logits shape: {logits.shape}")
    print(f"Val loss: {loss.item():.4f}")

    # Generate some text
    print("\n" + "=" * 60)
    print("Generating Sample Text")
    print("=" * 60)

    # Start with a single token (BOS or a random token)
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    print("\nGenerating 100 tokens...")
    generated = model.generate(context, max_new_tokens=100, temperature=0.8)
    generated_text = train_dataset.tokenizer.decode(generated[0].tolist())

    print("\nGenerated text:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)

    print("\n✓ All tests passed!\n")

# ============================================================================
# Main
# ============================================================================

def main():
    """Main function to load model and dataset."""
    print("\n" + "=" * 60)
    print("TOY TRANSFORMER MODEL AND DATA LOADER")
    print("=" * 60 + "\n")

    # Load model
    model, model_config = load_model()

    # Load dataset
    train_dataset, val_dataset = load_dataset(model_config)

    # Test everything works
    test_model_and_data(model, train_dataset, val_dataset, model_config)

    print("=" * 60)
    print("DONE!")
    print("=" * 60)

    return model, train_dataset, val_dataset

if __name__ == "__main__":
    model, train_dataset, val_dataset = main()

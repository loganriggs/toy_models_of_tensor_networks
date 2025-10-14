import sys
import os
# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ModelConfig, TrainingConfig, StreamingTextDataset, Trainer, ToyTransformer
import json
import torch
import argparse
import numpy as np

# Get the root directory (parent of scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Dataset Configuration
# ============================================================================
# Available datasets: 'fineweb', 'simplestories', 'tinystories'
DATASET_CONFIGS = {
    'fineweb': {
        'dataset_name': 'HuggingFaceFW/fineweb',
        'subset': 'sample-10BT',
        'tokenizer_name': 'gpt2',
        'vocab_size': 50257,
    },
    'simplestories': {
        'dataset_name': 'SimpleStories/SimpleStories',
        'subset': None,
        'tokenizer_name': 'SimpleStories/SimpleStories-1.25M',
        'vocab_size': 4096,
    },
    'tinystories': {
        'dataset_name': 'roneneldan/TinyStories',
        'subset': None,
        'tokenizer_name': 'roneneldan/TinyStories',
        'vocab_size': 10000,
    }
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train toy transformer models')
parser.add_argument('--dataset', type=str, default='simplestories',
                    choices=['fineweb', 'simplestories', 'tinystories'],
                    help='Dataset to use for training')
parser.add_argument('--model_type', type=str, default='attention_only_1L',
                    help='Model architecture type (e.g., transformer_10L, attention_only_2L)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=3e-3, help='Learning rate')
parser.add_argument('--num_batches', type=int, default=10000, help='Number of training batches')
parser.add_argument('--debug', action='store_true', help='Run in debug mode (100 batches)')
args = parser.parse_args()

# Override settings for debug mode
if args.debug:
    args.num_batches = 100
    eval_interval = 25
    log_interval = 10
    print("Running in DEBUG mode: 100 batches, eval_interval=25, log_interval=10")
else:
    eval_interval = 500
    log_interval = 50

# Get dataset configuration
dataset_config = DATASET_CONFIGS[args.dataset]
print(f"Using dataset: {args.dataset}")
print(f"  Dataset: {dataset_config['dataset_name']}")
print(f"  Tokenizer: {dataset_config['tokenizer_name']}")
print(f"  Vocab size: {dataset_config['vocab_size']}")

# Configure model
model_config = ModelConfig(
    model_type=args.model_type,
    vocab_size=dataset_config['vocab_size'],
    d_model=512,  # Moderate size for experiments
    n_head=8,
    n_ctx=512,  # Shorter context for faster training
    dropout=0.1
)

training_config = TrainingConfig(
    model_config=model_config,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    eval_interval=eval_interval,
    log_interval=log_interval
)

# Create data loaders
print("Initializing datasets...")
train_dataset = StreamingTextDataset(
    dataset_name=dataset_config['dataset_name'],
    subset=dataset_config['subset'],
    split='train',
    tokenizer_name=dataset_config['tokenizer_name'],
    seq_length=model_config.n_ctx,
    validation_ratio=0.001  # 0.1% for validation
)

val_dataset = StreamingTextDataset(
    dataset_name=dataset_config['dataset_name'],
    subset=dataset_config['subset'],
    split='validation',
    tokenizer_name=dataset_config['tokenizer_name'],
    seq_length=model_config.n_ctx,
    validation_ratio=0.001  # Same ratio to ensure consistent split
)


# Create model
model = ToyTransformer(model_config)
print(f"Model type: {model_config.model_type}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# Create trainer
trainer = Trainer(model, training_config)

# Training loop
print(f"Starting training for {args.num_batches} batches...")

# Track losses for plotting
train_losses = []
train_iters = []
val_losses = []
val_iters = []

for iter in range(args.num_batches):
    # Get batch of real data
    x, y = train_dataset.get_batch(training_config.batch_size)
    # Create attention mask (all ones since streaming data has no padding)
    attention_mask = torch.ones_like(x)

    # Train step
    loss, lr = trainer.train_step(x, attention_mask)

    # Track training loss
    train_losses.append(loss)
    train_iters.append(iter)

    # Logging
    if iter % training_config.log_interval == 0:
        print(f"Iter {iter}: loss={loss:.4f}, lr={lr:.6f}")
        
    # Evaluation
    if iter % training_config.eval_interval == 0 and iter > 0:
        val_losses_batch = []
        for _ in range(20):  # Evaluate on 20 batches
            x_val, y_val = val_dataset.get_batch(training_config.batch_size)
            attention_mask_val = torch.ones_like(x_val)
            _, val_loss = model(x_val, attention_mask_val)
            val_losses_batch.append(val_loss.item())
        val_loss_mean = np.mean(val_losses_batch)

        # Track validation loss
        val_losses.append(val_loss_mean)
        val_iters.append(iter)

        print(f"Validation loss: {val_loss_mean:.4f}")

        # Generate sample text
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device='cuda')
        generated = model.generate(context, max_new_tokens=100, temperature=0.8)
        print(f"Sample generation: {train_dataset.tokenizer.decode(generated[0].tolist())}")
        model.train()

print("Training complete!")

# Plot training curves
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_iters, train_losses, label='Training Loss', alpha=0.3)
# Smooth training loss with moving average
window = min(50, len(train_losses) // 10)
if window > 1:
    smooth_train = np.convolve(train_losses, np.ones(window)/window, mode='valid')
    smooth_iters = train_iters[window-1:]
    plt.plot(smooth_iters, smooth_train, label=f'Training Loss (smoothed, window={window})', linewidth=2)

if len(val_losses) > 0:
    plt.plot(val_iters, val_losses, 'o-', label='Validation Loss', linewidth=2, markersize=8)

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(f'Training Curves: {args.dataset} - {model_config.model_type} ({args.num_batches} batches)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot to figures folder
plot_filename = os.path.join(ROOT_DIR, 'figures', f'training_curves_{args.dataset}_{model_config.model_type}_{args.num_batches}batches.png')
plt.savefig(plot_filename, dpi=150)
print(f"Training curves saved to {plot_filename}")
plt.close()

# Save model with dataset name and batch count in filename
model_filename = os.path.join(ROOT_DIR, 'models', f'toy_transformer_{args.dataset}_{model_config.model_type}_{args.num_batches}batches.pt')
config_filename = os.path.join(ROOT_DIR, 'configs', f'{args.dataset}_{model_config.model_type}_{args.num_batches}batches_config.json')

torch.save(model.state_dict(), model_filename)
print(f"Model saved to {model_filename}")

# turn model config into a dictionary
model_config_dict = model_config.__dict__
# Add training info to config
training_info = {
    'num_batches': args.num_batches,
    'final_train_loss': train_losses[-1],
    'final_val_loss': val_losses[-1] if len(val_losses) > 0 else None,
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size
}
config_dict = {**model_config_dict, **training_info}

# save model config to a json file
with open(config_filename, 'w') as f:
    json.dump(config_dict, f, indent=2)
print(f"Config saved to {config_filename}")
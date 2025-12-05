import sys
import os
import time
# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ModelConfig, TrainingConfig, StreamingTextDataset, Trainer, ToyTransformer
from bilinear_mixer_transformer import BilinearMixerTransformer, BilinearMixerConfig
import json
import torch
import argparse
import numpy as np
import wandb

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
parser.add_argument('--model_type', type=str, default='transformer',
                    choices=['transformer', 'attention_only', 'embed_only', 'bilinear_mixer'],
                    help='Model architecture type')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Starting learning rate (linear decay to min_lr in second half)')
parser.add_argument('--num_batches', type=int, default=10000, help='Number of training batches')
parser.add_argument('--debug', action='store_true', help='Run in debug mode (100 batches)')
parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
parser.add_argument('--n_ctx', type=int, default=512, help='Context length / sequence length')
parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
parser.add_argument('--use_mixer', action='store_true', help='Use BilinearMixer instead of attention')
parser.add_argument('--mixer_r', type=int, default=None, help='Rank for BilinearMixer (defaults to n_head)')
parser.add_argument('--use_softmax', action='store_true', help='Use standard softmax attention instead of quadratic')
parser.add_argument('--use_bilinear_attn', action='store_true', help='Use CausalBilinearSelfAttention')
parser.add_argument('--bilinear_attn_qk_norm', action='store_true', help='Use RMS norm on QK matrices in CausalBilinearSelfAttention')
parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
parser.add_argument('--wandb_project', type=str, default='toy-transformers', help='Wandb project name')
parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name (auto-generated if not provided)')
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

# Configure model based on model_type
use_bilinear_mixer_transformer = args.model_type == 'bilinear_mixer'

if use_bilinear_mixer_transformer:
    # Use BilinearMixerTransformer with its own config
    model_config = BilinearMixerConfig(
        vocab_size=dataset_config['vocab_size'],
        d_model=args.d_model,
        n_ctx=args.n_ctx,
        n_heads=args.n_head,
        n_layers=args.n_layers,
        dropout=0.1,
    )
    seq_length = model_config.n_ctx
else:
    # Use ToyTransformer with ModelConfig
    model_config = ModelConfig(
        model_type=args.model_type,
        vocab_size=dataset_config['vocab_size'],
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layers,
        n_ctx=args.n_ctx,
        dropout=0.1,
        use_mixer=args.use_mixer,
        mixer_r=args.mixer_r,
        use_softmax=args.use_softmax,
        use_bilinear_attn=args.use_bilinear_attn,
        bilinear_attn_qk_norm=args.bilinear_attn_qk_norm
    )
    seq_length = model_config.n_ctx

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
    seq_length=seq_length,
    validation_ratio=0.001  # 0.1% for validation
)

val_dataset = StreamingTextDataset(
    dataset_name=dataset_config['dataset_name'],
    subset=dataset_config['subset'],
    split='validation',
    tokenizer_name=dataset_config['tokenizer_name'],
    seq_length=seq_length,
    validation_ratio=0.001  # Same ratio to ensure consistent split
)


# Create model
if use_bilinear_mixer_transformer:
    model = BilinearMixerTransformer(model_config)
    print(f"Model type: bilinear_mixer ({model_config.n_layers}L, {model_config.n_heads}H)")
    model_type_str = f'bilinear_mixer_{model_config.n_layers}L_{model_config.n_heads}H'
else:
    model = ToyTransformer(model_config)
    print(f"Model type: {model_config.model_type} ({model_config.n_layer}L)")
    model_type_str = f'{model_config.model_type}_{model_config.n_layer}L'
    if model_config.use_mixer:
        model_type_str += f'_mixer_r{model_config.mixer_r or model_config.n_head}'
    if model_config.use_bilinear_attn:
        model_type_str += '_bilinear_attn'

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Initialize wandb if enabled
if args.wandb:
    wandb_run_name = args.wandb_run_name or f'{args.dataset}_{model_type_str}_{args.num_batches}batches'
    wandb_config = {
        'dataset': args.dataset,
        'model_type': model_type_str,
        'd_model': args.d_model,
        'n_ctx': args.n_ctx,
        'n_head': args.n_head,
        'n_layers': args.n_layers,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_batches': args.num_batches,
        'num_params': num_params,
    }
    wandb.init(project=args.wandb_project, name=wandb_run_name, config=wandb_config)
    print(f"Wandb logging enabled: {args.wandb_project}/{wandb_run_name}")

# Compile model for faster training
print("Compiling model with torch.compile()...")
model = torch.compile(model)

# Create trainer
trainer = Trainer(model, training_config)
trainer.total_iterations = args.num_batches  # Set for LR warmdown schedule

# Training loop
print(f"Starting training for {args.num_batches} batches...")

# Track losses for plotting
train_losses = []
train_iters = []
val_losses = []
val_iters = []

# Track tokens and timing
tokens_per_batch = args.batch_size * seq_length
total_tokens = 0
start_time = time.time()
last_log_time = start_time

for iter in range(args.num_batches):
    # Get batch of real data
    x, y = train_dataset.get_batch(training_config.batch_size)
    # Create attention mask (all ones since streaming data has no padding)
    attention_mask = torch.ones_like(x)

    # Train step
    loss, lr, grad_norm = trainer.train_step(x, attention_mask)

    # Track training loss and tokens
    train_losses.append(loss)
    train_iters.append(iter)
    total_tokens += tokens_per_batch

    # Logging
    if iter % training_config.log_interval == 0:
        current_time = time.time()
        elapsed_since_last = current_time - last_log_time
        iters_per_sec = training_config.log_interval / elapsed_since_last if elapsed_since_last > 0 else 0
        last_log_time = current_time

        print(f"Iter {iter}: loss={loss:.4f}, lr={lr:.6f}, grad_norm={grad_norm:.4f}, it/s={iters_per_sec:.2f}, tokens={total_tokens:,}")
        if args.wandb:
            wandb.log({
                'train_loss': loss,
                'learning_rate': lr,
                'grad_norm': grad_norm,
                'iter': iter,
                'total_tokens': total_tokens,
                'iters_per_sec': iters_per_sec
            })
        
    # Evaluation
    if iter % training_config.eval_interval == 0 and iter > 0:
        val_losses_batch = []
        with torch.no_grad():
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
        if args.wandb:
            wandb.log({'val_loss': val_loss_mean, 'iter': iter})

        # Generate sample text
        # model.eval()
        # context = torch.zeros((1, 1), dtype=torch.long, device='cuda')
        # generated = model.generate(context, max_new_tokens=100, temperature=0.8)
        # print(f"Sample generation: {train_dataset.tokenizer.decode(generated[0].tolist())}")
        # model.train()

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

# Use pre-computed model_type_str for filename suffix
model_name_suffix = model_type_str
plot_title = f'Training Curves: {args.dataset} - {model_type_str} ({args.num_batches} batches)'

plt.title(plot_title)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot to figures folder
plot_filename = os.path.join(ROOT_DIR, 'figures', f'training_curves_{args.dataset}_{model_name_suffix}_{args.num_batches}batches.png')
plt.savefig(plot_filename, dpi=150)
print(f"Training curves saved to {plot_filename}")
plt.close()

# Save model with dataset name and batch count in filename
model_filename = os.path.join(ROOT_DIR, 'models', f'toy_transformer_{args.dataset}_{model_name_suffix}_{args.num_batches}batches.pt')
config_filename = os.path.join(ROOT_DIR, 'configs', f'{args.dataset}_{model_name_suffix}_{args.num_batches}batches_config.json')

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

# Finish wandb run
if args.wandb:
    wandb.finish()
    print("Wandb run finished")
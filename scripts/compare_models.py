# %% [markdown]
# # Model Comparison: CE Loss Analysis
# Compare embed-only, 1L mixer, and 2L mixer models on the same data points

# %%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import json
import numpy as np
from utils import ToyTransformer, ModelConfig, StreamingTextDataset
import matplotlib.pyplot as plt

# %%
# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths - adjust based on what's available
MODEL_CONFIGS = {
    'embed_only': {
        'model_path': 'models/toy_transformer_simplestories_embed_only_50000batches.pt',
        'config_path': 'configs/simplestories_embed_only_50000batches_config.json',
    },
    '1L_mixer_r1': {
        'model_path': 'models/toy_transformer_simplestories_attention_only_1L_mixer_r1_50000batches.pt',
        'config_path': 'configs/simplestories_attention_only_1L_mixer_r1_50000batches_config.json',
    },
    '1L_mixer_r2': {
        'model_path': 'models/toy_transformer_simplestories_attention_only_1L_mixer_r2_50000batches.pt',
        'config_path': 'configs/simplestories_attention_only_1L_mixer_r2_50000batches_config.json',
    },
    '2L_mixer_r1': {
        'model_path': 'models/toy_transformer_simplestories_attention_only_2L_mixer_r1_50000batches.pt',
        'config_path': 'configs/simplestories_attention_only_2L_mixer_r1_50000batches_config.json',
    },
    '2L_mixer_r2': {
        'model_path': 'models/toy_transformer_simplestories_attention_only_2L_mixer_r2_50000batches.pt',
        'config_path': 'configs/simplestories_attention_only_2L_mixer_r2_50000batches_config.json',
    },
}

# %%
def load_model(model_name, model_configs=MODEL_CONFIGS):
    """Load a model and its config"""
    config_info = model_configs[model_name]
    config_path = os.path.join(ROOT_DIR, config_info['config_path'])
    model_path = os.path.join(ROOT_DIR, config_info['model_path'])

    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return None, None
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None

    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Filter to only ModelConfig fields
    model_config_fields = ['vocab_size', 'd_model', 'n_ctx', 'n_head', 'dropout',
                           'model_type', 'use_mixer', 'mixer_r']
    filtered_config = {k: v for k, v in config_dict.items() if k in model_config_fields}
    config = ModelConfig(**filtered_config)

    # Load model
    model = ToyTransformer(config)
    state_dict = torch.load(model_path, map_location=device)

    # Handle torch.compile() prefix - strip "_orig_mod." from keys if present
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded {model_name}: {sum(p.numel() for p in model.parameters()):,} params")
    return model, config

# %%
# Load all available models
models = {}
configs = {}

for name in MODEL_CONFIGS.keys():
    model, config = load_model(name)
    if model is not None:
        models[name] = model
        configs[name] = config

print(f"\nLoaded {len(models)} models: {list(models.keys())}")

# %%
# Initialize dataset
print("\nInitializing dataset...")
dataset = StreamingTextDataset(
    dataset_name='SimpleStories/SimpleStories',
    subset=None,
    split='validation',
    tokenizer_name='SimpleStories/SimpleStories-1.25M',
    seq_length=128,
    validation_ratio=0.001
)

# %%
def compute_per_token_loss(model, input_ids, attention_mask=None):
    """Compute per-token CE loss for a batch"""
    with torch.no_grad():
        # Get logits
        x = model.embed(input_ids)
        x = model.dropout(x)

        for layer in model.layers:
            if hasattr(layer, 'forward'):
                # Check if layer accepts attention_mask
                try:
                    x = layer(x, attention_mask)
                except TypeError:
                    x = layer(x)

        logits = model.head(x)

        # Compute per-token loss
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Flatten and compute CE loss per token
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )

        # Reshape back to (batch, seq_len-1)
        loss_per_token = loss_per_token.view(input_ids.size(0), -1)

    return loss_per_token

# %%
def compute_batch_losses(models, input_ids, attention_mask=None):
    """Compute losses for all models on the same batch"""
    results = {}
    for name, model in models.items():
        loss_per_token = compute_per_token_loss(model, input_ids, attention_mask)
        results[name] = {
            'per_token': loss_per_token,
            'per_sample': loss_per_token.mean(dim=1),
            'mean': loss_per_token.mean().item()
        }
    return results

# %%
# Collect losses on validation data
num_batches = 50
batch_size = 32

all_losses = {name: [] for name in models.keys()}
all_per_sample_losses = {name: [] for name in models.keys()}
all_tokens = []

print(f"\nComputing losses on {num_batches} batches...")
for i in range(num_batches):
    x, _ = dataset.get_batch(batch_size, device=device)
    attention_mask = torch.ones_like(x)

    results = compute_batch_losses(models, x, attention_mask)

    for name in models.keys():
        all_losses[name].append(results[name]['mean'])
        all_per_sample_losses[name].append(results[name]['per_sample'].cpu())

    all_tokens.append(x.cpu())

    if (i + 1) % 10 == 0:
        print(f"  Batch {i+1}/{num_batches}")

# %%
# Aggregate results
print("\n" + "="*60)
print("MEAN VALIDATION LOSS BY MODEL")
print("="*60)

for name in sorted(models.keys()):
    mean_loss = np.mean(all_losses[name])
    print(f"{name:20s}: {mean_loss:.4f}")

# %%
# Concatenate per-sample losses
per_sample_losses = {name: torch.cat(all_per_sample_losses[name]).numpy()
                     for name in models.keys()}
tokens = torch.cat(all_tokens)

print(f"\nTotal samples evaluated: {len(per_sample_losses[list(models.keys())[0]])}")

# %%
# Plot loss distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of losses
ax = axes[0]
for name in sorted(models.keys()):
    ax.hist(per_sample_losses[name], bins=50, alpha=0.5, label=name, density=True)
ax.set_xlabel('Per-Sample Loss')
ax.set_ylabel('Density')
ax.set_title('Distribution of Per-Sample Losses')
ax.legend()
ax.grid(True, alpha=0.3)

# Box plot
ax = axes[1]
data = [per_sample_losses[name] for name in sorted(models.keys())]
ax.boxplot(data, labels=sorted(models.keys()))
ax.set_ylabel('Per-Sample Loss')
ax.set_title('Loss Distribution by Model')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, 'figures/model_loss_comparison.png'), dpi=150)
plt.show()

# %%
# Find samples where models differ most
if len(models) >= 2:
    model_names = sorted(models.keys())

    # Compare each model pair
    print("\n" + "="*60)
    print("SAMPLES WHERE MODELS DIFFER MOST")
    print("="*60)

    # Compare embed_only vs best mixer model
    if 'embed_only' in models and len(models) > 1:
        mixer_models = [n for n in model_names if n != 'embed_only']
        best_mixer = min(mixer_models, key=lambda n: np.mean(all_losses[n]))

        diff = per_sample_losses['embed_only'] - per_sample_losses[best_mixer]

        # Samples where mixer helps most (embed_only much worse)
        mixer_helps_idx = np.argsort(diff)[-10:][::-1]
        print(f"\nSamples where {best_mixer} helps most vs embed_only:")
        print(f"{'Idx':>6} | {'embed_only':>12} | {best_mixer:>12} | {'Diff':>8}")
        print("-" * 50)
        for idx in mixer_helps_idx:
            print(f"{idx:6d} | {per_sample_losses['embed_only'][idx]:12.4f} | "
                  f"{per_sample_losses[best_mixer][idx]:12.4f} | {diff[idx]:8.4f}")

        # Samples where embed_only is better (rare but interesting)
        embed_helps_idx = np.argsort(diff)[:10]
        print(f"\nSamples where embed_only is better than {best_mixer}:")
        print(f"{'Idx':>6} | {'embed_only':>12} | {best_mixer:>12} | {'Diff':>8}")
        print("-" * 50)
        for idx in embed_helps_idx:
            print(f"{idx:6d} | {per_sample_losses['embed_only'][idx]:12.4f} | "
                  f"{per_sample_losses[best_mixer][idx]:12.4f} | {diff[idx]:8.4f}")

# %%
# Compare 1L vs 2L models
if any('1L' in n for n in models.keys()) and any('2L' in n for n in models.keys()):
    print("\n" + "="*60)
    print("1L vs 2L MODEL COMPARISON")
    print("="*60)

    l1_models = [n for n in model_names if '1L' in n]
    l2_models = [n for n in model_names if '2L' in n]

    for l1, l2 in zip(sorted(l1_models), sorted(l2_models)):
        if l1.replace('1L', '2L') == l2:  # Matching r values
            diff = per_sample_losses[l1] - per_sample_losses[l2]
            mean_improvement = np.mean(diff)
            pct_2l_better = np.mean(diff > 0) * 100

            print(f"\n{l1} vs {l2}:")
            print(f"  Mean improvement (2L better): {mean_improvement:.4f}")
            print(f"  % samples where 2L is better: {pct_2l_better:.1f}%")

# %%
# Compare r=1 vs r=2
if any('r1' in n for n in models.keys()) and any('r2' in n for n in models.keys()):
    print("\n" + "="*60)
    print("r=1 vs r=2 COMPARISON")
    print("="*60)

    r1_models = [n for n in model_names if 'r1' in n]
    r2_models = [n for n in model_names if 'r2' in n]

    for r1 in sorted(r1_models):
        r2 = r1.replace('r1', 'r2')
        if r2 in models:
            diff = per_sample_losses[r1] - per_sample_losses[r2]
            mean_improvement = np.mean(diff)
            pct_r2_better = np.mean(diff > 0) * 100

            print(f"\n{r1} vs {r2}:")
            print(f"  Mean improvement (r2 better): {mean_improvement:.4f}")
            print(f"  % samples where r2 is better: {pct_r2_better:.1f}%")

# %%
# Correlation between model losses
print("\n" + "="*60)
print("LOSS CORRELATION BETWEEN MODELS")
print("="*60)

if len(models) >= 2:
    corr_matrix = np.zeros((len(models), len(models)))
    model_names = sorted(models.keys())

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            corr = np.corrcoef(per_sample_losses[name1], per_sample_losses[name2])[0, 1]
            corr_matrix[i, j] = corr

    print("\nCorrelation Matrix:")
    print(f"{'':20s}", end='')
    for name in model_names:
        print(f"{name:>15s}", end='')
    print()

    for i, name1 in enumerate(model_names):
        print(f"{name1:20s}", end='')
        for j, name2 in enumerate(model_names):
            print(f"{corr_matrix[i,j]:15.3f}", end='')
        print()

# %%
# Save detailed results
results_dict = {
    'mean_losses': {name: float(np.mean(all_losses[name])) for name in models.keys()},
    'std_losses': {name: float(np.std(per_sample_losses[name])) for name in models.keys()},
    'num_samples': len(per_sample_losses[list(models.keys())[0]]),
    'num_batches': num_batches,
    'batch_size': batch_size,
}

results_path = os.path.join(ROOT_DIR, 'outputs/model_comparison_results.json')
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"\nResults saved to {results_path}")

# %%
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

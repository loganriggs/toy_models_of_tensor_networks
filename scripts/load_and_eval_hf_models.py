# %% [markdown]
# # Load and Evaluate HuggingFace Models
#
# This notebook loads the bilinear attention models from HuggingFace
# and evaluates their CE loss on SimpleStories data.

# %%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from huggingface_hub import hf_hub_download
from utils import ToyTransformer, ModelConfig, StreamingTextDataset
import numpy as np

# %%
# Configuration
REPO_ID = "Elriggs/simplestories-bilinear-attn"
MODELS = [
    "attention_only_1L_bilinear_attn",
    "attention_only_2L_bilinear_attn",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_BATCHES = 50
BATCH_SIZE = 32

print(f"Device: {DEVICE}")

# %%
def load_model_from_hf(repo_id: str, model_name: str, device: str = "cuda"):
    """Load a model from HuggingFace Hub."""
    print(f"Loading {model_name} from {repo_id}...")

    # Download config
    config_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}/config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Store training info before removing
    training_info = {
        'num_batches': config_dict.pop('num_batches', None),
        'final_train_loss': config_dict.pop('final_train_loss', None),
        'final_val_loss': config_dict.pop('final_val_loss', None),
        'learning_rate': config_dict.pop('learning_rate', None),
        'batch_size': config_dict.pop('batch_size', None),
    }

    # Create model config and model
    config = ModelConfig(**config_dict)
    model = ToyTransformer(config)

    # Download and load weights
    weights_path = hf_hub_download(repo_id=repo_id, filename=f"{model_name}/pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {num_params:,} parameters")
    print(f"  Reported final val loss: {training_info['final_val_loss']:.4f}")

    return model, config, training_info

# %%
# Load all models
models = {}
for model_name in MODELS:
    model, config, training_info = load_model_from_hf(REPO_ID, model_name, DEVICE)
    models[model_name] = {
        'model': model,
        'config': config,
        'training_info': training_info,
    }

# %%
# Create dataset for evaluation
print("\nInitializing SimpleStories dataset...")
dataset = StreamingTextDataset(
    dataset_name='SimpleStories/SimpleStories',
    subset=None,
    split='validation',
    tokenizer_name='SimpleStories/SimpleStories-1.25M',
    seq_length=512,
    validation_ratio=0.001
)
print("Dataset ready.")

# %%
def evaluate_model(model, dataset, num_batches: int = 50, batch_size: int = 32, device: str = "cuda"):
    """Evaluate a model on the dataset, returning mean CE loss."""
    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_batch(batch_size, device=device)
            attention_mask = torch.ones_like(x)

            _, loss = model(x, attention_mask)
            losses.append(loss.item())

    return np.mean(losses), np.std(losses)

# %%
# Evaluate all models
print(f"\nEvaluating models on {NUM_EVAL_BATCHES} batches (batch_size={BATCH_SIZE})...")
print("=" * 60)

results = {}
for model_name, model_data in models.items():
    model = model_data['model']
    reported_loss = model_data['training_info']['final_val_loss']

    mean_loss, std_loss = evaluate_model(model, dataset, NUM_EVAL_BATCHES, BATCH_SIZE, DEVICE)
    results[model_name] = {'mean': mean_loss, 'std': std_loss}

    print(f"\n{model_name}:")
    print(f"  Evaluated CE Loss: {mean_loss:.4f} (+/- {std_loss:.4f})")
    print(f"  Reported Val Loss: {reported_loss:.4f}")
    print(f"  Difference: {abs(mean_loss - reported_loss):.4f}")

# %%
# Summary table
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Model':<40} {'Eval Loss':>10} {'Reported':>10}")
print("-" * 60)
for model_name, result in results.items():
    reported = models[model_name]['training_info']['final_val_loss']
    print(f"{model_name:<40} {result['mean']:>10.4f} {reported:>10.4f}")

# %%
# Sample inference - generate logits for a batch
print("\n" + "=" * 60)
print("SAMPLE INFERENCE")
print("=" * 60)

x, y = dataset.get_batch(4)
attention_mask = torch.ones_like(x)

for model_name, model_data in models.items():
    model = model_data['model']
    with torch.no_grad():
        logits, loss = model(x, attention_mask)

    print(f"\n{model_name}:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Batch loss: {loss.item():.4f}")

    # Show per-token loss for first sequence
    # Compute per-token CE loss
    ce_loss = torch.nn.functional.cross_entropy(
        logits[0, :-1].reshape(-1, logits.shape[-1]),
        y[0, 1:].reshape(-1),
        reduction='none'
    )
    print(f"  First sequence mean token loss: {ce_loss.mean().item():.4f}")
    print(f"  First sequence min/max token loss: {ce_loss.min().item():.2f} / {ce_loss.max().item():.2f}")

# %%
# print the first model
print(models[MODELS[0]]['model'])
print(models[MODELS[0]]['config'])
print(models[MODELS[0]]['training_info'])

# print the second model
print(models[MODELS[1]]['model'])
print(models[MODELS[1]]['config'])
print(models[MODELS[1]]['training_info'])
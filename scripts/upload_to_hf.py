"""
Script to upload trained ToyTransformer models to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --model_path models/toy_transformer_xxx.pt --config_path configs/xxx_config.json --model_name attention_only_1L_bilinear_attn

    # Or upload all bilinear_attn models:
    python scripts/upload_to_hf.py --upload_all_bilinear_attn
"""

import argparse
import torch
from huggingface_hub import HfApi, create_repo
import json
import os
import sys
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ToyTransformer, ModelConfig

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = "Elriggs/simplestories-bilinear-attn"


def upload_model_to_hf(model_path, config_path, model_name, repo_id=REPO_ID, token=None):
    """
    Upload a model checkpoint to Hugging Face Hub as a subfolder.
    """
    print(f"Loading model from {model_path}...")
    print(f"Loading config from {config_path}...")

    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Extract training info before creating ModelConfig
    training_info = {
        'num_batches': config_dict.pop('num_batches', None),
        'final_train_loss': config_dict.pop('final_train_loss', None),
        'final_val_loss': config_dict.pop('final_val_loss', None),
        'learning_rate': config_dict.pop('learning_rate', None),
        'batch_size': config_dict.pop('batch_size', None),
    }

    # Create model config
    config = ModelConfig(**config_dict)

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Remove _orig_mod. prefix from state_dict keys (from torch.compile)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Create and load model to verify weights
    model = ToyTransformer(config)
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded successfully: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create repository if needed
    api = HfApi(token=token)
    try:
        create_repo(repo_id, private=False, token=token, exist_ok=True)
        print(f"Created/verified repository: {repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return False

    # Create temp directory for this model
    temp_dir = os.path.join(ROOT_DIR, f"temp_hf_upload_{model_name}")
    os.makedirs(temp_dir, exist_ok=True)

    # Save model weights
    weights_path = os.path.join(temp_dir, "pytorch_model.bin")
    torch.save(state_dict, weights_path)
    print(f"Saved model weights to {weights_path}")

    # Save full config (including training info)
    full_config = {**config_dict, **training_info}
    config_save_path = os.path.join(temp_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(full_config, f, indent=2)
    print(f"Saved config to {config_save_path}")

    # Create README for this model
    readme_path = os.path.join(temp_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"""# {model_name}

ToyTransformer with CausalBilinearSelfAttention trained on SimpleStories.

## Model Config

- **Model type**: {config.model_type}
- **Layers**: {config.n_layer}
- **Heads**: {config.n_head}
- **d_model**: {config.d_model}
- **Context length**: {config.n_ctx}
- **Vocab size**: {config.vocab_size}
- **Bilinear attention**: {config.use_bilinear_attn}
- **QK norm**: {config.bilinear_attn_qk_norm}

## Training

- **Dataset**: SimpleStories
- **Batches**: {training_info['num_batches']:,}
- **Batch size**: {training_info['batch_size']}
- **Learning rate**: {training_info['learning_rate']}
- **Final train loss**: {training_info['final_train_loss']:.4f}
- **Final val loss**: {training_info['final_val_loss']:.4f}

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
import json
import sys
sys.path.insert(0, '/path/to/toy_models_of_tensor_networks')
from utils import ToyTransformer, ModelConfig

# Download config
config_path = hf_hub_download(repo_id="{repo_id}", filename="{model_name}/config.json")
with open(config_path) as f:
    config_dict = json.load(f)

# Remove training info fields
for key in ['num_batches', 'final_train_loss', 'final_val_loss', 'learning_rate', 'batch_size']:
    config_dict.pop(key, None)

# Create model
config = ModelConfig(**config_dict)
model = ToyTransformer(config)

# Download and load weights
weights_path = hf_hub_download(repo_id="{repo_id}", filename="{model_name}/pytorch_model.bin")
state_dict = torch.load(weights_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

model.eval()
```
""")
    print(f"Created README at {readme_path}")

    # Upload to subfolder
    print(f"Uploading {model_name} to Hugging Face Hub...")
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        path_in_repo=model_name,
        repo_type="model",
        token=token,
    )

    print(f"Model uploaded to https://huggingface.co/{repo_id}/tree/main/{model_name}")

    # Clean up
    shutil.rmtree(temp_dir)
    print("Cleaned up temporary files")

    return True


def upload_all_bilinear_attn(repo_id=REPO_ID, token=None):
    """Upload all bilinear attention models."""
    models_dir = os.path.join(ROOT_DIR, "models")
    configs_dir = os.path.join(ROOT_DIR, "configs")

    # Find all bilinear_attn models
    bilinear_models = []
    for f in os.listdir(models_dir):
        if 'bilinear_attn' in f and f.endswith('.pt'):
            bilinear_models.append(f)

    print(f"Found {len(bilinear_models)} bilinear attention models")

    for model_file in sorted(bilinear_models):
        # Parse model name from filename
        # Format: toy_transformer_simplestories_attention_only_1L_bilinear_attn_100000batches.pt
        parts = model_file.replace('.pt', '').split('_')
        # Extract the key part: attention_only_1L_bilinear_attn or similar
        idx = parts.index('simplestories') + 1
        model_name = '_'.join(parts[idx:]).replace('_100000batches', '')

        model_path = os.path.join(models_dir, model_file)
        config_file = model_file.replace('toy_transformer_', '').replace('.pt', '_config.json')
        config_path = os.path.join(configs_dir, config_file)

        if not os.path.exists(config_path):
            print(f"Warning: Config not found for {model_file}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Uploading: {model_name}")
        print(f"{'='*60}")

        upload_model_to_hf(model_path, config_path, model_name, repo_id, token)

    # Create main repo README
    create_repo_readme(repo_id, token)


def create_repo_readme(repo_id=REPO_ID, token=None):
    """Create main README for the repository."""
    api = HfApi(token=token)

    readme_content = """# SimpleStories Bilinear Attention Models

Collection of ToyTransformer models with CausalBilinearSelfAttention trained on SimpleStories dataset.

## Models

| Model | Layers | Type | Final Val Loss |
|-------|--------|------|----------------|
| `attention_only_1L_bilinear_attn` | 1 | Attention only | ~2.91 |
| `attention_only_2L_bilinear_attn` | 2 | Attention only | ~2.87 |

## Architecture

These models use **CausalBilinearSelfAttention** which computes squared attention patterns:
- Two QK pairs: `(Q1, K1)` and `(Q2, K2)`
- Attention scores: `(Q1 @ K1.T) * (Q2 @ K2.T)` (element-wise product)
- Causal masking applied
- RoPE (Rotary Positional Embeddings)

## Training Details

- **Dataset**: SimpleStories (4096 vocab)
- **d_model**: 256
- **n_head**: 4
- **n_ctx**: 512
- **Optimizer**: Muon
- **Batches**: 100,000
- **Batch size**: 128

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
import json

# Download a model
model_name = "attention_only_1L_bilinear_attn"
config_path = hf_hub_download(repo_id="Elriggs/simplestories-bilinear-attn", filename=f"{model_name}/config.json")
weights_path = hf_hub_download(repo_id="Elriggs/simplestories-bilinear-attn", filename=f"{model_name}/pytorch_model.bin")

# Load config
with open(config_path) as f:
    config_dict = json.load(f)

# Use with ToyTransformer from this repo
from utils import ToyTransformer, ModelConfig
for key in ['num_batches', 'final_train_loss', 'final_val_loss', 'learning_rate', 'batch_size']:
    config_dict.pop(key, None)
config = ModelConfig(**config_dict)
model = ToyTransformer(config)
model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
```

## Repository

Source code: [toy_models_of_tensor_networks](https://github.com/elriggs/toy_models_of_tensor_networks)
"""

    # Save and upload README
    temp_readme = os.path.join(ROOT_DIR, "temp_readme.md")
    with open(temp_readme, 'w') as f:
        f.write(readme_content)

    api.upload_file(
        path_or_fileobj=temp_readme,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    os.remove(temp_readme)
    print(f"\nMain README uploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload ToyTransformer models to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, help="Path to model .pt file")
    parser.add_argument("--config_path", type=str, help="Path to config .json file")
    parser.add_argument("--model_name", type=str, help="Name for the model in HF repo")
    parser.add_argument("--repo_id", type=str, default=REPO_ID, help="HF repository ID")
    parser.add_argument("--token", type=str, default=None, help="HF token (optional)")
    parser.add_argument("--upload_all_bilinear_attn", action="store_true", help="Upload all bilinear attention models")

    args = parser.parse_args()

    if args.upload_all_bilinear_attn:
        upload_all_bilinear_attn(args.repo_id, args.token)
    elif args.model_path and args.config_path and args.model_name:
        upload_model_to_hf(args.model_path, args.config_path, args.model_name, args.repo_id, args.token)
    else:
        parser.print_help()

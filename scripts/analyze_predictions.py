"""
Analyze model predictions: Find examples the model predicts well vs poorly
"""

import torch
import json
import argparse
import numpy as np
from utils import ToyTransformer, ModelConfig, StreamingTextDataset
from transformers import AutoTokenizer

def load_model(model_path, config_path, device='cuda'):
    """Load trained model from checkpoint"""
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Create model config (filter out training-specific keys)
    model_keys = ['vocab_size', 'd_model', 'n_ctx', 'n_head', 'dropout', 'model_type']
    model_config_dict = {k: v for k, v in config_dict.items() if k in model_keys}
    config = ModelConfig(**model_config_dict)

    # Load model
    model = ToyTransformer(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config, config_dict

def analyze_sample(model, tokens, attention_mask):
    """Get per-token losses for a sample"""
    with torch.no_grad():
        logits, total_loss = model(tokens, attention_mask)

    # Calculate per-token losses
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Per-token cross entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    per_token_losses = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    ).reshape(shift_labels.shape)

    # Apply mask and calculate mean loss per sequence
    masked_losses = per_token_losses * shift_mask
    seq_losses = masked_losses.sum(dim=1) / shift_mask.sum(dim=1)

    return seq_losses.cpu().numpy(), per_token_losses.cpu().numpy()

def get_predictions(model, tokenizer, sample_text, device='cuda'):
    """Get model predictions for a text sample"""
    # Tokenize
    tokens = tokenizer.encode(sample_text, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(tokens)

    with torch.no_grad():
        logits, loss = model(tokens, attention_mask)

    # Get predicted tokens
    predicted_ids = logits[0, :-1].argmax(dim=-1)
    actual_ids = tokens[0, 1:]

    # Calculate per-token losses
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    per_token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    return {
        'tokens': tokens[0].cpu().tolist(),
        'predicted_ids': predicted_ids.cpu().tolist(),
        'actual_ids': actual_ids.cpu().tolist(),
        'per_token_losses': per_token_losses.cpu().tolist(),
        'mean_loss': loss.item()
    }

def colorize_text_by_loss(tokenizer, tokens, losses, max_loss=3.0):
    """Create colored text visualization based on loss"""
    # ANSI color codes: green (good) to red (bad)
    def get_color(loss, max_loss):
        # Normalize loss to 0-1
        normalized = min(loss / max_loss, 1.0)
        if normalized < 0.33:
            return '\033[92m'  # Green
        elif normalized < 0.66:
            return '\033[93m'  # Yellow
        else:
            return '\033[91m'  # Red

    RESET = '\033[0m'

    colored_parts = []
    for i in range(1, len(tokens)):
        token_text = tokenizer.decode([tokens[i]])
        color = get_color(losses[i-1], max_loss)
        colored_parts.append(f"{color}{token_text}{RESET}")

    return ''.join(colored_parts)

def main():
    parser = argparse.ArgumentParser(description='Analyze model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pt file')
    parser.add_argument('--config', type=str, required=True, help='Path to config .json file')
    parser.add_argument('--dataset', type=str, default='simplestories',
                        choices=['simplestories', 'tinystories', 'fineweb'])
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to analyze')
    parser.add_argument('--show_best', type=int, default=5, help='Number of best predictions to show')
    parser.add_argument('--show_worst', type=int, default=5, help='Number of worst predictions to show')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Dataset config
    DATASET_CONFIGS = {
        'fineweb': {
            'dataset_name': 'HuggingFaceFW/fineweb',
            'subset': 'sample-10BT',
            'tokenizer_name': 'gpt2',
        },
        'simplestories': {
            'dataset_name': 'SimpleStories/SimpleStories',
            'subset': None,
            'tokenizer_name': 'SimpleStories/SimpleStories-1.25M',
        },
        'tinystories': {
            'dataset_name': 'roneneldan/TinyStories',
            'subset': None,
            'tokenizer_name': 'roneneldan/TinyStories',
        }
    }

    # Load model
    print(f"\nLoading model from {args.model}...")
    model, model_config, training_info = load_model(args.model, args.config, device)
    print(f"Model type: {model_config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if 'num_batches' in training_info:
        print(f"Trained for {training_info['num_batches']} batches")
        print(f"Final train loss: {training_info.get('final_train_loss', 'N/A'):.4f}")
        print(f"Final val loss: {training_info.get('final_val_loss', 'N/A')}")

    # Load tokenizer and dataset
    dataset_config = DATASET_CONFIGS[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(dataset_config['tokenizer_name'])
    tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading {args.dataset} validation data...")
    val_dataset = StreamingTextDataset(
        dataset_name=dataset_config['dataset_name'],
        subset=dataset_config['subset'],
        split='validation',
        tokenizer_name=dataset_config['tokenizer_name'],
        seq_length=model_config.n_ctx,
        validation_ratio=0.001
    )

    # Analyze samples
    print(f"\nAnalyzing {args.num_samples} samples...")
    all_losses = []
    all_samples = []

    for i in range(args.num_samples):
        if i % 100 == 0:
            print(f"  Processed {i}/{args.num_samples} samples...")

        # Get batch
        x, y = val_dataset.get_batch(1, device=device)
        attention_mask = torch.ones_like(x)

        # Analyze
        seq_losses, per_token_losses = analyze_sample(model, x, attention_mask)

        all_losses.append(seq_losses[0])
        all_samples.append({
            'tokens': x[0].cpu().tolist(),
            'loss': seq_losses[0],
            'per_token_losses': per_token_losses[0]
        })

    all_losses = np.array(all_losses)

    # Statistics
    print(f"\n{'='*70}")
    print("PREDICTION QUALITY STATISTICS")
    print(f"{'='*70}")
    print(f"Mean loss: {all_losses.mean():.4f}")
    print(f"Std loss: {all_losses.std():.4f}")
    print(f"Min loss: {all_losses.min():.4f}")
    print(f"Max loss: {all_losses.max():.4f}")
    print(f"Median loss: {np.median(all_losses):.4f}")

    # Find best and worst
    sorted_indices = np.argsort(all_losses)
    best_indices = sorted_indices[:args.show_best]
    worst_indices = sorted_indices[-args.show_worst:][::-1]

    # Show best predictions
    print(f"\n{'='*70}")
    print(f"TOP {args.show_best} BEST PREDICTIONS (Lowest Loss)")
    print(f"{'='*70}")
    for rank, idx in enumerate(best_indices, 1):
        sample = all_samples[idx]
        text = tokenizer.decode(sample['tokens'])
        colored_text = colorize_text_by_loss(tokenizer, sample['tokens'],
                                             sample['per_token_losses'], max_loss=3.0)

        print(f"\n[Rank {rank}] Loss: {sample['loss']:.4f}")
        print(f"Text: {text[:200]}...")
        print(f"Colored by loss: {colored_text[:500]}...")

    # Show worst predictions
    print(f"\n{'='*70}")
    print(f"TOP {args.show_worst} WORST PREDICTIONS (Highest Loss)")
    print(f"{'='*70}")
    for rank, idx in enumerate(worst_indices, 1):
        sample = all_samples[idx]
        text = tokenizer.decode(sample['tokens'])
        colored_text = colorize_text_by_loss(tokenizer, sample['tokens'],
                                             sample['per_token_losses'], max_loss=3.0)

        print(f"\n[Rank {rank}] Loss: {sample['loss']:.4f}")
        print(f"Text: {text[:200]}...")
        print(f"Colored by loss: {colored_text[:500]}...")

    # Show distribution
    print(f"\n{'='*70}")
    print("LOSS DISTRIBUTION")
    print(f"{'='*70}")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(all_losses, p)
        print(f"{p}th percentile: {val:.4f}")

if __name__ == '__main__':
    main()

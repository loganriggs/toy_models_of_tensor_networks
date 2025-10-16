"""
Fast teacher inference with full optimizations (compile + fast paths + AMP).

Pre-computes teacher model distributions on dataset samples and saves them
in HuggingFace dataset format for later use in mask training. This amortizes
the compilation overhead across many samples.

Usage:
    python scripts/fast_inference.py --n-samples 10000 --output teacher_distributions
    python scripts/fast_inference.py --n-samples 10000 --seed 42 --push-to-hub username/dataset-name
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
import os
import time
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ToyTransformer, ModelConfig, StreamingTextDataset


def get_dataset_samples_with_seed(dataset_name, tokenizer_name, n_samples,
                                   seq_length=512, device='cuda', seed=42):
    """Get n samples from dataset with deterministic seed"""
    from tokenization.tokenization import enc

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        split='train',
        tokenizer_name=tokenizer_name,
        seq_length=seq_length,
        subset=None,
        validation_ratio=0.001,
        seed=seed
    )

    samples = []
    print(f"Generating {n_samples} samples with seed={seed}...")

    for _ in tqdm(range(n_samples), desc="Loading samples"):
        x, _ = dataset.get_batch(batch_size=1, device=device)
        attention_mask = torch.ones_like(x)
        samples.append((x, attention_mask))

    return samples


def run_fast_inference(teacher_model, samples, device='cuda', batch_size=128):
    """Run optimized teacher inference and extract distributions

    Returns:
        results: List of dicts with {
            'input_ids': [seq_len],
            'attention_mask': [seq_len],
            'logits': [seq_len, vocab_size],
            'probs': [seq_len, vocab_size],
            'log_probs': [seq_len, vocab_size]
        }
    """
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("\n" + "="*80)
    print("COMPILING MODEL (this takes ~3-4 seconds, but will be amortized)")
    print("="*80)

    # Compile model with full optimizations
    print("Compiling with mode='max-autotune', fullgraph=True...")
    compiled_model = torch.compile(teacher_model, mode="max-autotune", fullgraph=True)

    # Warmup compilation with a sample
    print("Warming up compilation...")
    sample_data, sample_mask = samples[0]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        _ = compiled_model(sample_data, sample_mask)

    print("Compilation complete! Running inference...\n")

    # Run inference on all samples
    results = []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        for data, attention_mask in tqdm(samples, desc="Inference"):
            # Get teacher outputs
            teacher_output = compiled_model(data, attention_mask)
            logits = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output
            logits = logits.float()  # Convert to fp32 for storage

            # Compute distributions
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            # Store results (move to CPU to save GPU memory)
            results.append({
                'input_ids': data.squeeze(0).cpu().numpy(),  # [seq_len]
                'attention_mask': attention_mask.squeeze(0).cpu().numpy(),  # [seq_len]
                'logits': logits.squeeze(0).cpu().numpy(),  # [seq_len, vocab_size]
                'probs': probs.squeeze(0).cpu().numpy(),  # [seq_len, vocab_size]
                'log_probs': log_probs.squeeze(0).cpu().numpy(),  # [seq_len, vocab_size]
            })

    return results


def save_as_hf_dataset(results, output_path, metadata):
    """Save results as HuggingFace dataset

    Args:
        results: List of dicts from run_fast_inference
        output_path: Path to save dataset (directory name)
        metadata: Dict with model info, seed, etc.
    """
    try:
        from datasets import Dataset, Features, Value, Array2D, Array3D, Sequence
    except ImportError:
        print("ERROR: datasets library not installed. Install with: pip install datasets")
        return

    print("\n" + "="*80)
    print("SAVING AS HUGGINGFACE DATASET")
    print("="*80)

    # Convert list of dicts to dict of lists (HuggingFace format)
    data_dict = {
        'input_ids': [r['input_ids'] for r in results],
        'attention_mask': [r['attention_mask'] for r in results],
        'logits': [r['logits'] for r in results],
        'probs': [r['probs'] for r in results],
        'log_probs': [r['log_probs'] for r in results],
    }

    # Create dataset
    dataset = Dataset.from_dict(data_dict)

    # Add metadata as dataset info
    dataset.info.description = f"Teacher model distributions for {metadata['model_type']}"

    # Save dataset
    dataset.save_to_disk(output_path)
    print(f"✓ Saved dataset to: {output_path}")

    # Save metadata separately
    metadata_path = os.path.join(output_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")

    # Print dataset info
    print(f"\nDataset info:")
    print(f"  Num samples: {len(dataset)}")
    print(f"  Features: {dataset.features}")
    if dataset.dataset_size is not None:
        print(f"  Size on disk: {dataset.dataset_size / 1024**2:.1f} MB")

    return dataset


def load_hf_dataset(dataset_path):
    """Load pre-computed teacher distributions from disk

    Args:
        dataset_path: Path to saved dataset directory

    Returns:
        dataset: HuggingFace dataset
        metadata: Dict with model info, seed, etc.
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        print("ERROR: datasets library not installed. Install with: pip install datasets")
        return None, None

    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Load metadata
    metadata_path = os.path.join(dataset_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return dataset, metadata


def main():
    parser = argparse.ArgumentParser(description='Fast teacher inference and distribution saving')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of samples to generate (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for dataset generation (default: 42)')
    parser.add_argument('--output', type=str, default='data/teacher_distributions',
                       help='Output path for dataset (default: data/teacher_distributions)')
    parser.add_argument('--push-to-hub', type=str, default=None,
                       help='Push to HuggingFace Hub (format: username/dataset-name)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for inference (not used currently, samples are 1-by-1)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device != 'cuda':
        print("ERROR: CUDA not available. This script requires GPU.")
        return

    print("="*80)
    print("FAST TEACHER INFERENCE")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print(f"Output path: {args.output}")
    if args.push_to_hub:
        print(f"Will push to HuggingFace Hub: {args.push_to_hub}")
    print()

    # Load teacher model
    print("Loading teacher model...")
    model_path = 'models/toy_transformer_simplestories_transformer_3L_10000batches.pt'
    config_path = 'configs/simplestories_transformer_3L_10000batches_config.json'

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    valid_fields = ['vocab_size', 'd_model', 'n_ctx', 'n_head', 'dropout', 'model_type']
    config_dict_filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = ModelConfig(**config_dict_filtered)

    teacher_model = ToyTransformer(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    teacher_model.load_state_dict(state_dict)
    teacher_model.eval()

    # Disable dropout for deterministic inference
    teacher_model.dropout.p = 0.0
    for layer in teacher_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    print(f"✓ Model loaded: {config.model_type}")
    print(f"  d_model={config.d_model}, n_head={config.n_head}")
    print(f"  vocab_size={config.vocab_size}, n_ctx={config.n_ctx}")

    # Generate samples
    samples = get_dataset_samples_with_seed(
        dataset_name='SimpleStories/SimpleStories',
        tokenizer_name='SimpleStories/SimpleStories-1.25M',
        n_samples=args.n_samples,
        seq_length=config.n_ctx,
        device=device,
        seed=args.seed
    )

    print(f"✓ Generated {len(samples)} samples")
    print(f"  Sample shape: {samples[0][0].shape}")

    # Run fast inference
    start_time = time.time()
    results = run_fast_inference(teacher_model, samples, device=device)
    inference_time = time.time() - start_time

    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"Total time: {inference_time:.2f} seconds")
    print(f"Throughput: {args.n_samples / inference_time:.1f} samples/sec")
    print(f"Time per sample: {inference_time / args.n_samples * 1000:.2f} ms")

    # Calculate speedup vs baseline
    baseline_time_per_sample = 0.065  # From benchmark (baseline ~65ms per batch of 128)
    baseline_total_time = baseline_time_per_sample * args.n_samples / 128
    speedup = baseline_total_time / inference_time
    print(f"\nEstimated speedup vs baseline: {speedup:.2f}x")

    # Prepare metadata
    metadata = {
        'model_path': model_path,
        'config_path': config_path,
        'model_type': config.model_type,
        'd_model': config.d_model,
        'n_head': config.n_head,
        'vocab_size': config.vocab_size,
        'n_ctx': config.n_ctx,
        'n_samples': args.n_samples,
        'seed': args.seed,
        'dataset_name': 'SimpleStories/SimpleStories',
        'tokenizer_name': 'SimpleStories/SimpleStories-1.25M',
        'inference_time_sec': inference_time,
        'samples_per_sec': args.n_samples / inference_time,
        'optimizations': 'torch.compile(mode=max-autotune) + fast_paths + AMP',
        'amp_dtype': str(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'device': torch.cuda.get_device_name(0),
    }

    # Save as HuggingFace dataset
    dataset = save_as_hf_dataset(results, args.output, metadata)

    # Push to Hub if requested
    if args.push_to_hub and dataset is not None:
        print("\n" + "="*80)
        print("PUSHING TO HUGGINGFACE HUB")
        print("="*80)
        try:
            print(f"Pushing to: {args.push_to_hub}")
            dataset.push_to_hub(args.push_to_hub, private=True)
            print(f"✓ Successfully pushed to HuggingFace Hub!")
            print(f"  View at: https://huggingface.co/datasets/{args.push_to_hub}")
        except Exception as e:
            print(f"✗ Error pushing to Hub: {e}")
            print("Make sure you're logged in: huggingface-cli login")

    print("\n" + "="*80)
    print("USAGE EXAMPLE")
    print("="*80)
    print("\nTo load this dataset in your training script:")
    print(f"""
from datasets import load_from_disk
import json

# Load dataset
dataset = load_from_disk('{args.output}')

# Load metadata
with open('{args.output}/metadata.json', 'r') as f:
    metadata = json.load(f)

# Access samples
for i in range(len(dataset)):
    input_ids = dataset[i]['input_ids']  # [seq_len]
    logits = dataset[i]['logits']  # [seq_len, vocab_size]
    probs = dataset[i]['probs']  # [seq_len, vocab_size]
    log_probs = dataset[i]['log_probs']  # [seq_len, vocab_size]
    # Use for KL divergence training...
""")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

"""
Verify that optimized teacher outputs are acceptable for KL divergence training.

Tests whether Fast paths + AMP produce similar enough distributions for KL training.
"""

import torch
import torch.nn.functional as F
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ToyTransformer, ModelConfig


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device != 'cuda':
        print("CUDA not available. This requires CUDA.")
        return

    print("="*80)
    print("TEACHER OUTPUT VERIFICATION FOR KL DIVERGENCE TRAINING")
    print("="*80)

    # Load teacher model
    print("\nLoading teacher model...")
    model_path = 'models/toy_transformer_simplestories_transformer_3L_10000batches.pt'
    config_path = 'configs/simplestories_transformer_3L_10000batches_config.json'

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    valid_fields = ['vocab_size', 'd_model', 'n_ctx', 'n_head', 'dropout', 'model_type']
    config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = ModelConfig(**config_dict)

    teacher_model = ToyTransformer(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    teacher_model.load_state_dict(state_dict)
    teacher_model.eval()

    # Disable dropout
    teacher_model.dropout.p = 0.0
    for layer in teacher_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    print(f"Model: {config.model_type}, d_model={config.d_model}\n")

    # Create test data
    batch_size = 128
    seq_len = 256
    data = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("Testing: FAST PATHS + AMP (recommended configuration)")
    print("-" * 80)

    # Baseline output (FP32, no optimizations)
    with torch.no_grad():
        baseline_output = teacher_model(data, attention_mask)
        baseline_logits = baseline_output[0] if isinstance(baseline_output, tuple) else baseline_output

    # Optimized output (Fast paths + AMP)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            opt_output = teacher_model(data, attention_mask)
            opt_logits = opt_output[0] if isinstance(opt_output, tuple) else opt_output
            opt_logits = opt_logits.float()  # Convert back to fp32 for comparison

    # Logit-level comparison
    print("\n1. LOGIT-LEVEL DIFFERENCES:")
    logit_diff = (baseline_logits - opt_logits).abs()
    print(f"   Max absolute difference:  {logit_diff.max().item():.4f}")
    print(f"   Mean absolute difference: {logit_diff.mean().item():.6f}")
    print(f"   Median absolute difference: {logit_diff.median().item():.6f}")

    # Sample for percentile (full tensor too large)
    logit_diff_sample = logit_diff.flatten()[::100]  # Sample every 100th element
    print(f"   95th percentile (sampled): {torch.quantile(logit_diff_sample, 0.95).item():.6f}")

    # Distribution comparison (what actually matters for KL divergence)
    print("\n2. PROBABILITY DISTRIBUTION DIFFERENCES:")
    baseline_probs = F.softmax(baseline_logits, dim=-1)
    opt_probs = F.softmax(opt_logits, dim=-1)

    prob_diff = (baseline_probs - opt_probs).abs()
    print(f"   Max probability difference:  {prob_diff.max().item():.6f}")
    print(f"   Mean probability difference: {prob_diff.mean().item():.8f}")
    print(f"   Median probability difference: {prob_diff.median().item():.8f}")

    # KL divergence between distributions
    print("\n3. KL DIVERGENCE (what we actually optimize):")
    # KL(baseline || optimized) - how much information is lost
    kl_forward = F.kl_div(
        F.log_softmax(opt_logits, dim=-1),
        F.log_softmax(baseline_logits, dim=-1),
        log_target=True,
        reduction='none'
    ).sum(dim=-1)  # [B, S]

    print(f"   Mean KL divergence per token:   {kl_forward.mean().item():.8f}")
    print(f"   Median KL divergence per token: {kl_forward.median().item():.8f}")
    print(f"   Max KL divergence per token:    {kl_forward.max().item():.8f}")

    # Sample for percentile
    kl_sample = kl_forward.flatten()[::10]  # Sample every 10th element
    print(f"   95th percentile (sampled):      {torch.quantile(kl_sample, 0.95).item():.8f}")

    # Top-k prediction agreement
    print("\n4. TOP-K PREDICTION AGREEMENT:")
    for k in [1, 5, 10]:
        baseline_topk = baseline_logits.topk(k, dim=-1).indices
        opt_topk = opt_logits.topk(k, dim=-1).indices

        # Calculate overlap
        matches = 0
        total = baseline_topk.numel()
        for b in range(baseline_topk.shape[0]):
            for s in range(baseline_topk.shape[1]):
                baseline_set = set(baseline_topk[b, s].cpu().tolist())
                opt_set = set(opt_topk[b, s].cpu().tolist())
                matches += len(baseline_set & opt_set)

        agreement = matches / total * 100
        print(f"   Top-{k} agreement: {agreement:.2f}%")

    # Cross-entropy loss comparison (simulating student training)
    print("\n5. CROSS-ENTROPY LOSS (simulating student training):")
    # Simulate random student predictions
    student_logits = torch.randn_like(baseline_logits)

    ce_baseline = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        baseline_probs.view(-1, baseline_probs.size(-1)),
        reduction='none'
    ).view(baseline_logits.shape[:2])

    ce_optimized = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        opt_probs.view(-1, opt_probs.size(-1)),
        reduction='none'
    ).view(opt_logits.shape[:2])

    ce_diff = (ce_baseline - ce_optimized).abs()
    print(f"   Mean CE difference:   {ce_diff.mean().item():.6f}")
    print(f"   Median CE difference: {ce_diff.median().item():.6f}")
    print(f"   Max CE difference:    {ce_diff.max().item():.6f}")

    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    kl_mean = kl_forward.mean().item()
    prob_mean = prob_diff.mean().item()
    top1_agreement = matches / total * 100  # Last computed was top-10

    # Recompute top-1 for verdict
    baseline_top1 = baseline_logits.argmax(dim=-1)
    opt_top1 = opt_logits.argmax(dim=-1)
    top1_agreement = (baseline_top1 == opt_top1).float().mean().item() * 100

    print(f"\nKey metrics:")
    print(f"  - Mean KL divergence per token: {kl_mean:.8f}")
    print(f"  - Mean probability difference:  {prob_mean:.8f}")
    print(f"  - Top-1 prediction agreement:   {top1_agreement:.2f}%")

    if kl_mean < 1e-4 and top1_agreement > 99.0:
        print(f"\n✓ EXCELLENT: Distributions are nearly identical.")
        print(f"  Safe to use for KL divergence training.")
    elif kl_mean < 1e-3 and top1_agreement > 95.0:
        print(f"\n✓ GOOD: Distributions are very similar.")
        print(f"  Safe to use for KL divergence training.")
    elif kl_mean < 0.01:
        print(f"\n⚠ ACCEPTABLE: Small differences exist but unlikely to affect training.")
        print(f"  Probably safe to use for KL divergence training.")
    else:
        print(f"\n✗ CONCERNING: Significant differences detected.")
        print(f"  May affect training quality - verify empirically.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

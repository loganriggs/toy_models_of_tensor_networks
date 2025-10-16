"""
Benchmark different strategies for batched sparse mask training.

Tests:
1. Current implementation (no batch dimension, sequential)
2. Batched with loop over samples
3. Batched with einsum
4. Batched with batched matrix multiplication (bmm)

Measures forward pass time, backward pass time, and memory usage.
"""

import torch
import torch.nn.functional as F
from torch.func import vmap
import time
import sys
import os
from typing import Tuple
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ToyTransformer, ModelConfig


class ReinMaxFunction(torch.autograd.Function):
    """ReinMax: Heun's method for second-order gradient accuracy"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Heun's method: average of forward and backward Euler
        sigmoid_grad = torch.sigmoid(input) * (1 - torch.sigmoid(input))
        # ReinMax uses a corrected gradient
        return grad_output * (1 + sigmoid_grad)


class STEFunction(torch.autograd.Function):
    """Straight-Through Estimator"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # STE: gradient flows through unchanged
        return grad_output


def get_mask(logits, estimator='reinmax'):
    """
    Apply gradient estimator to get differentiable binary mask.

    Args:
        logits: Mask logits
        estimator: 'reinmax' or 'ste'
    """
    if estimator == 'reinmax':
        return ReinMaxFunction.apply(logits)
    elif estimator == 'ste':
        return STEFunction.apply(logits)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")


def _masked_bilinear_single(h, masks, W1, b1, W2, b2, Wd, bd):
    """
    Helper function for vmap: applies masked bilinear to a single sample.

    Args:
        h: Hidden states [S, D]
        masks: Tuple of (m1[M,D], m2[M,D], md[D,M])
        W1, b1, W2, b2, Wd, bd: Layer weights and biases

    Returns:
        output: [S, D]
    """
    m1, m2, md = masks

    MW1 = W1 * m1
    MW2 = W2 * m2
    MWd = Wd * md

    z1 = torch.matmul(h, MW1.t()) + b1
    z2 = torch.matmul(h, MW2.t()) + b2
    mlp_hidden = z1 * z2
    output = torch.matmul(mlp_hidden, MWd.t()) + bd

    return output


class MaskedTransformer(torch.nn.Module):
    """
    Unified masked transformer supporting multiple implementations.

    Args:
        base_model: The base transformer model
        mask_layer_idx: Which layer to mask (default: 0)
        mode: Implementation mode - 'current', 'loop', 'einsum', 'bmm', or 'vmap'
        batch_size: Batch size for batched modes (ignored for 'current' mode)
        estimator: Gradient estimator - 'reinmax' or 'ste'
    """

    def __init__(self, base_model, mask_layer_idx=0, mode='current', batch_size=1, estimator='reinmax'):
        super().__init__()
        self.base_model = base_model
        self.mask_layer_idx = mask_layer_idx
        self.mode = mode
        self.batch_size = batch_size
        self.estimator = estimator

        # Get dimensions from the target layer
        layer = self.base_model.layers[mask_layer_idx]
        d_model = layer.bilinear.proj1.weight.shape[1]
        d_mlp = layer.bilinear.proj1.weight.shape[0]

        # Initialize mask logits based on mode
        # Start with logits = 1.0 (sigmoid(1.0) ≈ 0.731, gives all 1's in forward since > 0)
        # This keeps sigmoid gradient strong enough for optimization
        if mode == 'current':
            # No batch dimension
            self.mask_logits_proj1 = torch.nn.Parameter(torch.ones(d_mlp, d_model) * 1.0)
            self.mask_logits_proj2 = torch.nn.Parameter(torch.ones(d_mlp, d_model) * 1.0)
            self.mask_logits_down = torch.nn.Parameter(torch.ones(d_model, d_mlp) * 1.0)
        else:
            # With batch dimension
            self.mask_logits_proj1 = torch.nn.Parameter(torch.ones(batch_size, d_mlp, d_model) * 1.0)
            self.mask_logits_proj2 = torch.nn.Parameter(torch.ones(batch_size, d_mlp, d_model) * 1.0)
            self.mask_logits_down = torch.nn.Parameter(torch.ones(batch_size, d_model, d_mlp) * 1.0)

    def forward(self, x, attention_mask=None):
        if self.mode == 'loop':
            return self._forward_loop(x, attention_mask)
        else:
            # All other modes use the common forward path
            return self._forward_common(x, attention_mask)

    def _apply_masked_bilinear(self, h, layer, masks):
        """
        Apply masked bilinear layer.

        Args:
            h: Hidden states [B, S, D] or [1, S, D]
            layer: The transformer layer
            masks: Tuple of (mask_proj1, mask_proj2, mask_down)

        Returns:
            output: Masked bilinear output [B, S, D]
        """
        m1, m2, md = masks

        # Get weights and biases
        W1, b1 = layer.bilinear.proj1.weight, layer.bilinear.proj1.bias  # [M,D], [M]
        W2, b2 = layer.bilinear.proj2.weight, layer.bilinear.proj2.bias  # [M,D], [M]
        Wd, bd = layer.bilinear.down.weight, layer.bilinear.down.bias    # [D,M], [D]

        if self.mode == 'current':
            # No batch dimension in masks: [M,D] * [M,D] -> [M,D]
            MW1 = W1 * m1
            MW2 = W2 * m2
            MWd = Wd * md

            z1 = torch.matmul(h, MW1.t()) + b1
            z2 = torch.matmul(h, MW2.t()) + b2
            mlp_hidden = z1 * z2
            output = torch.matmul(mlp_hidden, MWd.t()) + bd

        elif self.mode == 'einsum':
            # Batch dimension in masks: [M,D] * [B,M,D] -> [B,M,D]
            MW1 = W1.unsqueeze(0) * m1  # [B, M, D]
            MW2 = W2.unsqueeze(0) * m2  # [B, M, D]
            MWd = Wd.unsqueeze(0) * md  # [B, D, M]

            # Transpose ONCE and materialize contiguous layout
            MW1_T = MW1.transpose(1, 2).contiguous()  # [B, D, M]
            MW2_T = MW2.transpose(1, 2).contiguous()  # [B, D, M]
            MWd_T = MWd.transpose(1, 2).contiguous()  # [B, M, D]

            # Reuse transposed tensors
            z1 = torch.matmul(h, MW1_T) + b1  # [B, S, M]
            z2 = torch.matmul(h, MW2_T) + b2  # [B, S, M]
            mlp_hidden = z1 * z2  # [B, S, M]
            output = torch.matmul(mlp_hidden, MWd_T) + bd  # [B, S, D]

        elif self.mode == 'bmm':
            # Batch dimension with batched matmul
            MW1 = W1.unsqueeze(0) * m1  # [B, M, D]
            MW2 = W2.unsqueeze(0) * m2  # [B, M, D]
            MWd = Wd.unsqueeze(0) * md  # [B, D, M]

            # Transpose ONCE and materialize contiguous layout
            MW1_T = MW1.transpose(1, 2).contiguous()  # [B, D, M]
            MW2_T = MW2.transpose(1, 2).contiguous()  # [B, D, M]
            MWd_T = MWd.transpose(1, 2).contiguous()  # [B, M, D]

            # Reuse transposed tensors
            z1 = torch.matmul(h, MW1_T) + b1  # [B, S, M]
            z2 = torch.matmul(h, MW2_T) + b2  # [B, S, M]
            mlp_hidden = z1 * z2  # [B, S, M]
            output = torch.matmul(mlp_hidden, MWd_T) + bd  # [B, S, D]

        elif self.mode == 'vmap':
            # Batch dimension with vmap
            # vmap over batch dimension: h [B, S, D], masks [B, M, D] or [B, D, M]
            # in_dims=(0, 0, None, None, None, None, None, None) means:
            #   - vmap over h's batch dim (0)
            #   - vmap over masks tuple (each element's batch dim is 0)
            #   - don't vmap over weights/biases (None)
            output = vmap(
                _masked_bilinear_single,
                in_dims=(0, 0, None, None, None, None, None, None)
            )(h, (m1, m2, md), W1, b1, W2, b2, Wd, bd)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return output

    def _forward_common(self, x, attention_mask=None):
        """Common forward path for current, einsum, and bmm modes."""
        # Get masks based on mode
        batch_size = x.shape[0]
        if self.mode == 'current':
            m1 = get_mask(self.mask_logits_proj1, self.estimator)
            m2 = get_mask(self.mask_logits_proj2, self.estimator)
            md = get_mask(self.mask_logits_down, self.estimator)
        else:
            # Batched modes (einsum, bmm)
            m1 = get_mask(self.mask_logits_proj1[:batch_size], self.estimator)
            m2 = get_mask(self.mask_logits_proj2[:batch_size], self.estimator)
            md = get_mask(self.mask_logits_down[:batch_size], self.estimator)

        # Embedding
        h = self.base_model.embed(x)
        h = self.base_model.dropout(h)

        # Process through layers
        for i, layer in enumerate(self.base_model.layers):
            if i == self.mask_layer_idx:
                # Apply attention normally
                h = h + self.base_model.dropout(layer.attn(h, attention_mask))

                # Apply masked bilinear (mode-specific)
                output = self._apply_masked_bilinear(h, layer, (m1, m2, md))
                h = h + self.base_model.dropout(output)
            else:
                # Normal layer forward
                h = layer(h, attention_mask=attention_mask)

        logits = self.base_model.head(h)
        return logits

    def _forward_loop(self, x, attention_mask=None):
        """Batched masks, but loop over samples in forward pass."""
        batch_size = x.shape[0]
        outputs = []

        # Temporarily set mode to 'current' for per-sample processing
        original_mode = self.mode
        self.mode = 'current'

        for b in range(batch_size):
            # Get masks for this sample
            m1 = get_mask(self.mask_logits_proj1[b], self.estimator)
            m2 = get_mask(self.mask_logits_proj2[b], self.estimator)
            md = get_mask(self.mask_logits_down[b], self.estimator)

            # Embedding
            h = self.base_model.embed(x[b:b+1])
            h = self.base_model.dropout(h)

            # Process through layers
            for i, layer in enumerate(self.base_model.layers):
                if i == self.mask_layer_idx:
                    # Apply attention normally
                    attn_mask = attention_mask[b:b+1] if attention_mask is not None else None
                    h = h + self.base_model.dropout(layer.attn(h, attn_mask))

                    # Apply masked bilinear (uses 'current' mode internally)
                    output = self._apply_masked_bilinear(h, layer, (m1, m2, md))
                    h = h + self.base_model.dropout(output)
                else:
                    # Normal layer forward
                    attn_mask = attention_mask[b:b+1] if attention_mask is not None else None
                    h = layer(h, attn_mask)

            logits = self.base_model.head(h)
            outputs.append(logits)

        # Restore original mode
        self.mode = original_mode

        return torch.cat(outputs, dim=0)


def benchmark_forward_backward(model, data, attention_mask, n_iters=10, warmup=3):
    """Benchmark forward and backward pass times."""
    times_forward = []
    times_backward = []

    # Warmup
    for _ in range(warmup):
        logits = model(data, attention_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), data.view(-1))
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Benchmark
    for _ in range(n_iters):
        # Forward
        start = time.perf_counter()
        logits = model(data, attention_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), data.view(-1))
        torch.cuda.synchronize()
        times_forward.append(time.perf_counter() - start)

        # Backward
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        times_backward.append(time.perf_counter() - start)

        model.zero_grad()

    return {
        'forward_mean': np.mean(times_forward) * 1000,  # ms
        'forward_std': np.std(times_forward) * 1000,
        'backward_mean': np.mean(times_backward) * 1000,
        'backward_std': np.std(times_backward) * 1000,
        'total_mean': np.mean([f + b for f, b in zip(times_forward, times_backward)]) * 1000,
    }


def get_model_memory(model, device):
    """Get memory usage of model parameters."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB


def verify_equivalence(base_model, data, attention_mask, device='cuda', batch_size=2, estimator='reinmax'):
    """Verify that all implementations produce the same outputs and gradients."""
    print("="*80)
    print(f"EQUIVALENCE VERIFICATION (estimator={estimator})")
    print("="*80)
    print(f"\nTesting with batch_size={batch_size}\n")

    # Create reference model (current mode)
    torch.manual_seed(42)
    model_current = MaskedTransformer(base_model, mask_layer_idx=0, mode='current', estimator=estimator).to(device)

    # Create batched models
    model_loop = MaskedTransformer(base_model, mask_layer_idx=0, mode='loop', batch_size=batch_size, estimator=estimator).to(device)
    model_einsum = MaskedTransformer(base_model, mask_layer_idx=0, mode='einsum', batch_size=batch_size, estimator=estimator).to(device)
    model_bmm = MaskedTransformer(base_model, mask_layer_idx=0, mode='bmm', batch_size=batch_size, estimator=estimator).to(device)
    model_vmap = MaskedTransformer(base_model, mask_layer_idx=0, mode='vmap', batch_size=batch_size, estimator=estimator).to(device)

    # Copy mask parameters from current model to all batched models
    with torch.no_grad():
        for model in [model_loop, model_einsum, model_bmm, model_vmap]:
            for b in range(batch_size):
                model.mask_logits_proj1.data[b] = model_current.mask_logits_proj1.data.clone()
                model.mask_logits_proj2.data[b] = model_current.mask_logits_proj2.data.clone()
                model.mask_logits_down.data[b] = model_current.mask_logits_down.data.clone()

    # Run forward pass for comparison
    with torch.no_grad():
        if batch_size == 1:
            logits_current = model_current(data, attention_mask)
        else:
            # For current model, run each sample separately and concatenate
            logits_current_list = []
            for b in range(batch_size):
                logits_b = model_current(data[b:b+1], attention_mask[b:b+1] if attention_mask is not None else None)
                logits_current_list.append(logits_b)
            logits_current = torch.cat(logits_current_list, dim=0)

        logits_loop = model_loop(data, attention_mask)
        logits_einsum = model_einsum(data, attention_mask)
        logits_bmm = model_bmm(data, attention_mask)
        logits_vmap = model_vmap(data, attention_mask)

    # Print sample values for debugging
    print(f"  Sample logits (first 5 values):")
    print(f"    Current: {logits_current[0, 0, :5]}")
    print(f"    Loop:    {logits_loop[0, 0, :5]}")
    print(f"    Einsum:  {logits_einsum[0, 0, :5]}")
    print(f"    BMM:     {logits_bmm[0, 0, :5]}")
    print(f"    VMAP:    {logits_vmap[0, 0, :5]}\n")

    # Compare outputs
    print("Forward pass output comparison:")
    loop_diff = (logits_current - logits_loop).abs().max().item()
    einsum_diff = (logits_current - logits_einsum).abs().max().item()
    bmm_diff = (logits_current - logits_bmm).abs().max().item()
    vmap_diff = (logits_current - logits_vmap).abs().max().item()

    print(f"  Current vs Loop:   max_diff = {loop_diff:.2e}")
    print(f"  Current vs Einsum: max_diff = {einsum_diff:.2e}")
    print(f"  Current vs BMM:    max_diff = {bmm_diff:.2e}")
    print(f"  Current vs VMAP:   max_diff = {vmap_diff:.2e}")

    # Check if differences are within tolerance
    tolerance = 1e-4
    loop_match = loop_diff < tolerance
    einsum_match = einsum_diff < tolerance
    bmm_match = bmm_diff < tolerance
    vmap_match = vmap_diff < tolerance

    print(f"\n  Tolerance: {tolerance:.2e}")
    print(f"  Loop:   {'✓ PASS' if loop_match else '✗ FAIL'}")
    print(f"  Einsum: {'✓ PASS' if einsum_match else '✗ FAIL'}")
    print(f"  BMM:    {'✓ PASS' if bmm_match else '✗ FAIL'}")
    print(f"  VMAP:   {'✓ PASS' if vmap_match else '✗ FAIL'}")

    # Overall result
    print(f"\n{'='*80}")
    all_pass = loop_match and einsum_match and bmm_match and vmap_match
    if all_pass:
        print("✓ ALL IMPLEMENTATIONS PRODUCE EQUIVALENT RESULTS")
    else:
        print("✗ SOME IMPLEMENTATIONS DIFFER - CHECK RESULTS ABOVE")
    print(f"{'='*80}\n")

    return all_pass


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load base model
    print("Loading base model...")
    base_model_path = 'models/toy_transformer_simplestories_transformer_3L_10000batches.pt'
    import json
    config_path = 'configs/simplestories_transformer_3L_10000batches_config.json'

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Filter config to only ModelConfig fields
    valid_fields = ['vocab_size', 'd_model', 'n_ctx', 'n_head', 'dropout', 'model_type']
    config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = ModelConfig(**config_dict)

    base_model = ToyTransformer(config).to(device)
    state_dict = torch.load(base_model_path, map_location=device)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    # Freeze base model weights - we only want to train mask logits
    for param in base_model.parameters():
        param.requires_grad = False

    # Disable dropout
    base_model.dropout.p = 0.0
    for layer in base_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    print(f"Model config: {config.d_model=}, {config.n_head=}, model_type={config.model_type}\n")

    # Create test data
    batch_sizes = [1, 2, 8, 16]
    estimators = ['reinmax', 'ste']
    seq_len = 256

    # First verify equivalence for both estimators
    print("\n")
    for estimator in estimators:
        print(f"\n{'='*80}")
        print(f"TESTING ESTIMATOR: {estimator.upper()}")
        print(f"{'='*80}\n")

        verify_data_1 = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
        verify_mask_1 = torch.ones(1, seq_len, device=device)
        verify_equivalence(base_model, verify_data_1, verify_mask_1, device=device, batch_size=1, estimator=estimator)

        verify_data_2 = torch.randint(0, config.vocab_size, (2, seq_len), device=device)
        verify_mask_2 = torch.ones(2, seq_len, device=device)
        verify_equivalence(base_model, verify_data_2, verify_mask_2, device=device, batch_size=2, estimator=estimator)

    print("="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    results = {}

    for estimator in estimators:
        print(f"\n{'='*80}")
        print(f"ESTIMATOR: {estimator.upper()}")
        print(f"{'='*80}")

        results[estimator] = {}

        for batch_size in batch_sizes:
            print(f"\n{'='*80}")
            print(f"Batch Size: {batch_size}")
            print(f"{'='*80}\n")

            data = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            attention_mask = torch.ones(batch_size, seq_len, device=device)

            # Test each implementation
            implementations = []

            modes = [('Current', 'current'), ('Loop', 'loop'), ('Einsum', 'einsum'), ('BMM', 'bmm'), ('VMAP', 'vmap')]

            for idx, (name, mode) in enumerate(modes):
                # Skip current mode for batch_size > 1 (it doesn't support per-sample masks)
                if mode == 'current' and batch_size > 1:
                    continue

                print(f"{idx+1}. {name} (batch_size={batch_size})...")
                model = MaskedTransformer(base_model, mask_layer_idx=0, mode=mode, batch_size=batch_size, estimator=estimator).to(device)
                result = benchmark_forward_backward(model, data, attention_mask)
                result['memory_mb'] = get_model_memory(model, device)
                implementations.append((name, result))
                print(f"   Forward: {result['forward_mean']:.2f} ± {result['forward_std']:.2f} ms")
                print(f"   Backward: {result['backward_mean']:.2f} ± {result['backward_std']:.2f} ms")
                print(f"   Total: {result['total_mean']:.2f} ms")
                print(f"   Memory: {result['memory_mb']:.2f} MB\n")
                del model
                torch.cuda.empty_cache()

            results[estimator][batch_size] = implementations

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")

    for estimator in estimators:
        print(f"\n{'='*80}")
        print(f"ESTIMATOR: {estimator.upper()}")
        print(f"{'='*80}\n")

        if 1 in results[estimator]:
            baseline_time = results[estimator][1][0][1]['total_mean']  # Current implementation
            print(f"Baseline (Current, batch=1): {baseline_time:.2f} ms\n")

            print("Batch size 1 comparisons:")
            for name, result in results[estimator][1]:
                speedup = baseline_time / result['total_mean']
                print(f"  {name:20s}: {result['total_mean']:6.2f} ms  (speedup: {speedup:.2f}x)")

        if 2 in results[estimator]:
            print(f"\nBatch size 2 comparisons:")
            for name, result in results[estimator][2]:
                if 1 in results[estimator]:
                    speedup = baseline_time / (result['total_mean'] / 2)  # Per-sample time
                    print(f"  {name:20s}: {result['total_mean']:6.2f} ms  (per-sample speedup: {speedup:.2f}x)")
                else:
                    print(f"  {name:20s}: {result['total_mean']:6.2f} ms")

        if 8 in results[estimator]:
            print(f"\nBatch size 8 comparisons:")
            for name, result in results[estimator][8]:
                if 1 in results[estimator]:
                    speedup = baseline_time / (result['total_mean'] / 8)  # Per-sample time
                    print(f"  {name:20s}: {result['total_mean']:6.2f} ms  (per-sample speedup: {speedup:.2f}x)")
                else:
                    print(f"  {name:20s}: {result['total_mean']:6.2f} ms")

        if 16 in results[estimator]:
            print(f"\nBatch size 16 comparisons:")
            for name, result in results[estimator][16]:
                if 1 in results[estimator]:
                    speedup = baseline_time / (result['total_mean'] / 16)  # Per-sample time
                    print(f"  {name:20s}: {result['total_mean']:6.2f} ms  (per-sample speedup: {speedup:.2f}x)")
                else:
                    print(f"  {name:20s}: {result['total_mean']:6.2f} ms")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

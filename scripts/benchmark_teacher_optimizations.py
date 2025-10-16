"""
Benchmark teacher model inference with various PyTorch optimizations.

Tests different optimization strategies:
1. Baseline (no optimizations)
2. Fast paths (TF32, high precision matmul, expandable segments)
3. torch.compile (max-autotune)
4. AMP (Automatic Mixed Precision)
5. CUDA graphs
6. Combinations of the above

Runs 10k samples at batch_size=128 to measure realistic throughput.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
import json
import numpy as np
from contextlib import nullcontext

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ToyTransformer, ModelConfig


def benchmark_inference(model, data, attention_mask, n_batches=100, warmup=10,
                       use_amp=False, amp_dtype=torch.float16, cuda_graph=None):
    """
    Benchmark model inference.

    Args:
        model: Model to benchmark
        data: Input data [B, S]
        attention_mask: Attention mask [B, S]
        n_batches: Number of batches to run
        warmup: Number of warmup iterations
        use_amp: Whether to use automatic mixed precision
        amp_dtype: dtype for AMP (bfloat16 or float16)
        cuda_graph: Pre-captured CUDA graph (if using)

    Returns:
        dict with timing statistics
    """
    times = []

    # Setup AMP context
    amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if cuda_graph is not None:
                cuda_graph.replay()
            else:
                with amp_context:
                    _ = model(data, attention_mask)

    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        for _ in range(n_batches):
            start = time.perf_counter()

            if cuda_graph is not None:
                cuda_graph.replay()
            else:
                with amp_context:
                    logits = model(data, attention_mask)

            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'total_sec': np.sum(times),
    }


def verify_equivalence(logits_ref, logits_test, tolerance=1e-2, name="Test"):
    """Verify that two outputs are close enough."""
    max_diff = (logits_ref - logits_test).abs().max().item()
    mean_diff = (logits_ref - logits_test).abs().mean().item()

    passed = max_diff < tolerance
    status = "✓ PASS" if passed else "✗ FAIL"

    print(f"  {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} ... {status}")
    return passed


def capture_cuda_graph(model, static_data, static_mask, use_amp=False, amp_dtype=torch.float16):
    """Capture a CUDA graph for the model."""
    print("  Capturing CUDA graph...")

    # Warmup
    amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    with torch.no_grad():
        for _ in range(3):
            with amp_context:
                _ = model(static_data, static_mask)

    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            with amp_context:
                static_output = model(static_data, static_mask)

    return graph, static_output


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device != 'cuda':
        print("CUDA not available. This benchmark requires CUDA.")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}\n")

    # Load teacher model
    print("Loading teacher model...")
    model_path = 'models/toy_transformer_simplestories_transformer_3L_10000batches.pt'
    config_path = 'configs/simplestories_transformer_3L_10000batches_config.json'

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    valid_fields = ['vocab_size', 'd_model', 'n_ctx', 'n_head', 'dropout', 'model_type']
    config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = ModelConfig(**config_dict)

    base_model = ToyTransformer(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    # Disable dropout
    base_model.dropout.p = 0.0
    for layer in base_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    print(f"Model: {config.model_type}, d_model={config.d_model}, n_head={config.n_head}\n")

    # Test configuration
    batch_size = 128
    seq_len = 256
    total_samples = 10000
    n_batches = total_samples // batch_size  # ~78 batches

    print(f"Benchmark configuration:")
    print(f"  Total samples: {total_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of batches: {n_batches}\n")

    # Create test data
    data = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Determine AMP dtype
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using AMP dtype: {amp_dtype}\n")

    print("="*80)
    print("OPTIMIZATION BENCHMARKS")
    print("="*80)

    results = {}
    reference_output = None

    # 1. Baseline (no optimizations)
    print("\n1. BASELINE (no optimizations)")
    print("-" * 40)
    model1 = base_model
    with torch.no_grad():
        output = model1(data, attention_mask)
        # Handle tuple returns (logits, ...) - extract just logits
        reference_output = output[0] if isinstance(output, tuple) else output
    result1 = benchmark_inference(model1, data, attention_mask, n_batches=n_batches)
    results['baseline'] = result1
    print(f"  Time per batch: {result1['mean_ms']:.2f} ± {result1['std_ms']:.2f} ms")
    print(f"  Total time for {total_samples} samples: {result1['total_sec']:.2f} sec")
    print(f"  Throughput: {total_samples/result1['total_sec']:.1f} samples/sec")

    # 2. Fast paths (TF32, high precision)
    print("\n2. FAST PATHS (TF32 + high precision matmul)")
    print("-" * 40)

    # Save original settings
    orig_tf32 = torch.backends.cuda.matmul.allow_tf32
    orig_cudnn_tf32 = torch.backends.cudnn.allow_tf32

    # Enable fast paths
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    model2 = base_model
    with torch.no_grad():
        output = model2(data, attention_mask)
        output2 = output[0] if isinstance(output, tuple) else output
    verify_equivalence(reference_output, output2, tolerance=1e-2, name="Fast paths")
    result2 = benchmark_inference(model2, data, attention_mask, n_batches=n_batches)
    results['fast_paths'] = result2
    print(f"  Time per batch: {result2['mean_ms']:.2f} ± {result2['std_ms']:.2f} ms")
    print(f"  Total time: {result2['total_sec']:.2f} sec")
    print(f"  Throughput: {total_samples/result2['total_sec']:.1f} samples/sec")
    print(f"  Speedup: {result1['total_sec']/result2['total_sec']:.2f}x")

    # 3. torch.compile (max-autotune)
    print("\n3. TORCH.COMPILE (mode='max-autotune')")
    print("-" * 40)
    print("  Compiling model (this may take a few minutes)...")
    model3_compiled = torch.compile(base_model, mode="max-autotune", fullgraph=True)

    # Warmup compilation
    with torch.no_grad():
        for _ in range(3):
            _ = model3_compiled(data, attention_mask)

    with torch.no_grad():
        output = model3_compiled(data, attention_mask)
        output3_compiled = output[0] if isinstance(output, tuple) else output
    verify_equivalence(reference_output, output3_compiled, tolerance=1e-2, name="torch.compile")
    result3_compiled = benchmark_inference(model3_compiled, data, attention_mask, n_batches=n_batches)
    results['torch_compile'] = result3_compiled
    print(f"  Time per batch: {result3_compiled['mean_ms']:.2f} ± {result3_compiled['std_ms']:.2f} ms")
    print(f"  Total time: {result3_compiled['total_sec']:.2f} sec")
    print(f"  Throughput: {total_samples/result3_compiled['total_sec']:.1f} samples/sec")
    print(f"  Speedup: {result1['total_sec']/result3_compiled['total_sec']:.2f}x")

    # 4. AMP (Automatic Mixed Precision)
    print("\n4. AMP (Automatic Mixed Precision)")
    print("-" * 40)
    model4 = base_model
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = model4(data, attention_mask)
            output4 = output[0] if isinstance(output, tuple) else output
    verify_equivalence(reference_output, output4.float(), tolerance=5e-2, name="AMP")
    result4 = benchmark_inference(model4, data, attention_mask, n_batches=n_batches,
                                  use_amp=True, amp_dtype=amp_dtype)
    results['amp'] = result4
    print(f"  Time per batch: {result4['mean_ms']:.2f} ± {result4['std_ms']:.2f} ms")
    print(f"  Total time: {result4['total_sec']:.2f} sec")
    print(f"  Throughput: {total_samples/result4['total_sec']:.1f} samples/sec")
    print(f"  Speedup: {result1['total_sec']/result4['total_sec']:.2f}x")

    # 5. Fast paths + AMP
    print("\n5. FAST PATHS + AMP")
    print("-" * 40)
    model5 = base_model
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = model5(data, attention_mask)
            output5 = output[0] if isinstance(output, tuple) else output
    verify_equivalence(reference_output, output5.float(), tolerance=5e-2, name="Fast+AMP")
    result5 = benchmark_inference(model5, data, attention_mask, n_batches=n_batches,
                                  use_amp=True, amp_dtype=amp_dtype)
    results['fast_amp'] = result5
    print(f"  Time per batch: {result5['mean_ms']:.2f} ± {result5['std_ms']:.2f} ms")
    print(f"  Total time: {result5['total_sec']:.2f} sec")
    print(f"  Throughput: {total_samples/result5['total_sec']:.1f} samples/sec")
    print(f"  Speedup: {result1['total_sec']/result5['total_sec']:.2f}x")

    # 6. CUDA Graph (with fast paths + AMP)
    print("\n6. CUDA GRAPH (with Fast Paths + AMP)")
    print("-" * 40)
    model6 = base_model

    # Create static buffers
    static_data = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    static_mask = torch.empty((batch_size, seq_len), dtype=torch.float32, device=device)
    static_data.copy_(data)
    static_mask.copy_(attention_mask)

    graph6, static_output = capture_cuda_graph(model6, static_data, static_mask,
                                               use_amp=True, amp_dtype=amp_dtype)
    static_output6 = static_output[0] if isinstance(static_output, tuple) else static_output
    verify_equivalence(reference_output, static_output6.float(), tolerance=5e-2, name="CUDA Graph")
    result6 = benchmark_inference(model6, static_data, static_mask, n_batches=n_batches,
                                  use_amp=True, amp_dtype=amp_dtype, cuda_graph=graph6)
    results['cuda_graph'] = result6
    print(f"  Time per batch: {result6['mean_ms']:.2f} ± {result6['std_ms']:.2f} ms")
    print(f"  Total time: {result6['total_sec']:.2f} sec")
    print(f"  Throughput: {total_samples/result6['total_sec']:.1f} samples/sec")
    print(f"  Speedup: {result1['total_sec']/result6['total_sec']:.2f}x")

    # 7. torch.compile + Fast paths + AMP (FULL STACK)
    print("\n7. TORCH.COMPILE + FAST PATHS + AMP (full stack)")
    print("-" * 40)
    print("  Compiling model with fast paths enabled (this may take a few minutes)...")

    # Fast paths are still enabled from earlier
    model7_compiled = torch.compile(base_model, mode="max-autotune", fullgraph=True)

    # Warmup compilation with AMP
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            for _ in range(3):
                _ = model7_compiled(data, attention_mask)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = model7_compiled(data, attention_mask)
            output7 = output[0] if isinstance(output, tuple) else output
    verify_equivalence(reference_output, output7.float(), tolerance=5e-2, name="Compile+Fast+AMP")
    result7 = benchmark_inference(model7_compiled, data, attention_mask, n_batches=n_batches,
                                  use_amp=True, amp_dtype=amp_dtype)
    results['compile_fast_amp'] = result7
    print(f"  Time per batch: {result7['mean_ms']:.2f} ± {result7['std_ms']:.2f} ms")
    print(f"  Total time: {result7['total_sec']:.2f} sec")
    print(f"  Throughput: {total_samples/result7['total_sec']:.1f} samples/sec")
    print(f"  Speedup: {result1['total_sec']/result7['total_sec']:.2f}x")

    # Restore original settings
    torch.backends.cuda.matmul.allow_tf32 = orig_tf32
    torch.backends.cudnn.allow_tf32 = orig_cudnn_tf32

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nProcessing {total_samples} samples:\n")

    for name, result in results.items():
        speedup = result1['total_sec'] / result['total_sec']
        throughput = total_samples / result['total_sec']
        print(f"  {name:15s}: {result['total_sec']:6.2f} sec  "
              f"({throughput:6.1f} samples/sec)  "
              f"[{speedup:.2f}x speedup]")

    # Best configuration
    best_name = min(results.keys(), key=lambda k: results[k]['total_sec'])
    best_speedup = result1['total_sec'] / results[best_name]['total_sec']

    print(f"\nBest configuration: {best_name.upper()}")
    print(f"  Total speedup: {best_speedup:.2f}x")
    print(f"  Time saved: {result1['total_sec'] - results[best_name]['total_sec']:.2f} sec")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

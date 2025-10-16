"""
Measure torch.compile compilation overhead.
"""

import torch
import time
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

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # Load teacher model
    print("Loading teacher model...")
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

    print(f"Model loaded: {config.model_type}, d_model={config.d_model}\n")

    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Create test data
    batch_size = 128
    seq_len = 256
    data = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    print("="*80)
    print("COMPILATION TIME MEASUREMENT")
    print("="*80)

    # Measure compilation time
    print("\nCompiling model (mode='max-autotune', fullgraph=True)...")
    print("This includes the first forward pass that triggers compilation.\n")

    start_compile = time.perf_counter()

    # Compile
    compiled_model = torch.compile(teacher_model, mode="max-autotune", fullgraph=True)

    # First forward pass triggers compilation
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _ = compiled_model(data, attention_mask)

    torch.cuda.synchronize()
    compile_time = time.perf_counter() - start_compile

    print(f"Total compilation time: {compile_time:.2f} seconds")
    print(f"                       ({compile_time/60:.2f} minutes)")

    # Additional warmup passes (should be fast now)
    print("\nRunning 5 warmup passes to verify compilation is complete...")
    warmup_times = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            for i in range(5):
                start = time.perf_counter()
                _ = compiled_model(data, attention_mask)
                torch.cuda.synchronize()
                warmup_times.append(time.perf_counter() - start)

    print(f"Warmup pass times: {[f'{t*1000:.1f}ms' for t in warmup_times]}")
    print(f"Average warmup time: {sum(warmup_times)/len(warmup_times)*1000:.2f} ms")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOne-time compilation cost: {compile_time:.2f} seconds")
    print(f"Per-batch inference time: {sum(warmup_times)/len(warmup_times)*1000:.2f} ms")
    print(f"\nFor a 5-second mask training run:")
    print(f"  - Compilation overhead: {compile_time:.2f}s")
    print(f"  - Total time with compilation: {compile_time + 5:.2f}s")
    print(f"  - Overhead percentage: {compile_time/(compile_time+5)*100:.1f}%")

    # Calculate break-even point
    # Without compile: teacher inference at baseline ~65ms/batch
    # With compile: ~13.5ms/batch + compilation overhead
    baseline_time_per_batch = 0.065  # seconds
    compiled_time_per_batch = sum(warmup_times)/len(warmup_times)
    speedup = baseline_time_per_batch / compiled_time_per_batch

    # Break even: compile_time = (baseline_time_per_batch - compiled_time_per_batch) * n_batches
    breakeven_batches = compile_time / (baseline_time_per_batch - compiled_time_per_batch)
    breakeven_samples = breakeven_batches * batch_size

    print(f"\nBreak-even analysis:")
    print(f"  - Speedup per batch: {speedup:.2f}x")
    print(f"  - Break-even point: {breakeven_batches:.0f} batches ({breakeven_samples:.0f} samples)")
    print(f"  - If processing <{breakeven_samples:.0f} samples: don't compile")
    print(f"  - If processing >{breakeven_samples:.0f} samples: compile is faster")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

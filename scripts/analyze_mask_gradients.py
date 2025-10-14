"""
Analyze gradients of mask logits on first forward pass.

Loads trained masks and examines gradient structure when masks are initialized to 0
(all weights masked out). This helps understand what the gradient signal looks like
for learning sparsity patterns.
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
from pathlib import Path

# Setup paths relative to script location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add parent directory to path
sys.path.append(str(PROJECT_ROOT))
from utils import ToyTransformer, ModelConfig, StreamingTextDataset

# Change to project root so relative paths work
os.chdir(PROJECT_ROOT)
print(f"Working directory: {os.getcwd()}")

# %%
# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = PROJECT_ROOT / 'models' / 'toy_transformer_simplestories_transformer_3L_10000batches.pt'
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'simplestories_transformer_3L_10000batches_config.json'
MASK_WEIGHTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = PROJECT_ROOT / 'figures'
DATA_SEED = 42  # Same seed used in training

# Create output directories if they don't exist
MASK_WEIGHTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Model path: {MODEL_PATH}")
print(f"Config path: {CONFIG_PATH}")

# %%
# Load teacher model

def load_model(model_path, config_path, device='cuda'):
    """Load a trained transformer model"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = ModelConfig(
        vocab_size=config_dict['vocab_size'],
        d_model=config_dict['d_model'],
        n_ctx=config_dict['n_ctx'],
        n_head=config_dict['n_head'],
        dropout=config_dict.get('dropout', 0.1),
        model_type=config_dict['model_type']
    )

    model = ToyTransformer(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config

print("Loading teacher model...")
teacher_model, model_config = load_model(str(MODEL_PATH), str(CONFIG_PATH), device=DEVICE)
print(f"Model loaded: {model_config.model_type}")
print(f"  d_model={model_config.d_model}, n_head={model_config.n_head}")
print(f"  vocab_size={model_config.vocab_size}, n_ctx={model_config.n_ctx}")

# Disable dropout
teacher_model.dropout.p = 0.0
for layer in teacher_model.layers:
    if hasattr(layer, 'dropout'):
        layer.dropout.p = 0.0

# %%
# Load dataset samples with same seed

def get_dataset_samples(dataset_name, tokenizer_name, n_samples=10,
                       seq_length=512, device='cuda', seed=42):
    """Get n separate samples from the dataset with fixed seed for reproducibility"""
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    for _ in range(n_samples):
        x, _ = dataset.get_batch(batch_size=1, device=device)
        attention_mask = torch.ones_like(x)
        samples.append((x, attention_mask))

    return samples

print("Loading dataset samples...")
n_samples = 5  # Analyze first 5 samples
samples = get_dataset_samples(
    dataset_name='SimpleStories/SimpleStories',
    tokenizer_name='SimpleStories/SimpleStories-1.25M',
    n_samples=n_samples,
    seq_length=model_config.n_ctx,
    device=DEVICE,
    seed=DATA_SEED
)
print(f"Loaded {len(samples)} samples")

# %%
# Define MaskedTransformer class (copy from training script)

class ReinMaxFunction(torch.autograd.Function):
    """ReinMax: Heun's method for second-order gradient accuracy"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid_grad = torch.sigmoid(input) * (1 - torch.sigmoid(input))
        return grad_output * (1 + sigmoid_grad)


class MaskedTransformer(nn.Module):
    """Wraps a trained transformer and adds learnable masks to layer 0's bilinear weights"""

    def __init__(self, base_model, mask_layer_idx=0, init_value=1.0):
        super().__init__()
        self.base_model = base_model
        self.mask_layer_idx = mask_layer_idx

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get the bilinear layer
        bilinear = self.base_model.layers[mask_layer_idx].bilinear

        # Initialize mask parameters
        self.mask_logits_proj1 = nn.Parameter(
            torch.ones_like(bilinear.proj1.weight) * init_value
        )
        self.mask_logits_proj2 = nn.Parameter(
            torch.ones_like(bilinear.proj2.weight) * init_value
        )
        self.mask_logits_down = nn.Parameter(
            torch.ones_like(bilinear.down.weight) * init_value
        )

    def get_masks(self):
        """Apply ReinMax to get binary masks"""
        mask_proj1 = ReinMaxFunction.apply(self.mask_logits_proj1)
        mask_proj2 = ReinMaxFunction.apply(self.mask_logits_proj2)
        mask_down = ReinMaxFunction.apply(self.mask_logits_down)
        return mask_proj1, mask_proj2, mask_down

    def forward(self, x, attention_mask=None, use_masks=True):
        """Forward pass with masked bilinear layer"""
        mask_proj1, mask_proj2, mask_down = self.get_masks()

        x_emb = self.base_model.embed(x)
        x_emb = self.base_model.dropout(x_emb)

        for i, layer in enumerate(self.base_model.layers):
            if i == self.mask_layer_idx and use_masks:
                # Apply attention normally
                x_emb = x_emb + self.base_model.dropout(layer.attn(x_emb, attention_mask))

                # Apply bilinear with masked weights
                bilinear = layer.bilinear
                masked_proj1_weight = bilinear.proj1.weight * mask_proj1
                masked_proj2_weight = bilinear.proj2.weight * mask_proj2
                masked_down_weight = bilinear.down.weight * mask_down

                hidden1 = F.linear(x_emb, masked_proj1_weight, bilinear.proj1.bias)
                hidden2 = F.linear(x_emb, masked_proj2_weight, bilinear.proj2.bias)
                hidden = hidden1 * hidden2
                output = F.linear(hidden, masked_down_weight, bilinear.down.bias)

                x_emb = x_emb + self.base_model.dropout(output)
            else:
                if hasattr(layer, 'attn'):
                    x_emb = layer(x_emb, attention_mask)
                else:
                    x_emb = layer(x_emb, attention_mask)

        logits = self.base_model.head(x_emb)
        return logits

# %%
# Function to compute gradients with masks initialized to 0

def compute_initial_gradients(teacher_model, data, attention_mask, device='cuda', init_value=0.1):
    """
    Compute gradients for mask logits initialized to 0 (all weights masked out).
    Returns gradients for proj1, proj2, and down masks.
    """
    # Create masked model with logits initialized to 0
    masked_model = MaskedTransformer(teacher_model, mask_layer_idx=0, init_value=init_value).to(device)

    # Disable dropout
    masked_model.base_model.dropout.p = 0.0
    for layer in masked_model.base_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    masked_model.train()

    # Get teacher predictions
    with torch.no_grad():
        teacher_logits = teacher_model(data, attention_mask)[0]
        teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Forward pass with masked model
    student_logits = masked_model(data, attention_mask)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # KL divergence loss
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    # Backward pass
    kl_loss.backward()

    # Extract gradients
    grad_proj1 = masked_model.mask_logits_proj1.grad.clone()
    grad_proj2 = masked_model.mask_logits_proj2.grad.clone()
    grad_down = masked_model.mask_logits_down.grad.clone()

    return {
        'grad_proj1': grad_proj1,
        'grad_proj2': grad_proj2,
        'grad_down': grad_down,
        'kl_loss': kl_loss.item()
    }

# %%
# UTILITY: Evaluate a mask configuration

def evaluate_mask_configuration(teacher_model, mask_logits_proj1, mask_logits_proj2,
                               mask_logits_down, data, attention_mask, device='cuda'):
    """
    Evaluate KL divergence and CE diff for a given mask configuration.

    Args:
        teacher_model: Original teacher model
        mask_logits_proj1, mask_logits_proj2, mask_logits_down: Mask logits (tensors)
        data, attention_mask: Input data
        device: Device to run on

    Returns:
        Dictionary with metrics
    """
    # Create masked model
    masked_model = MaskedTransformer(teacher_model, mask_layer_idx=0, init_value=1.0).to(device)

    # Load the provided mask logits
    masked_model.mask_logits_proj1.data = mask_logits_proj1.to(device)
    masked_model.mask_logits_proj2.data = mask_logits_proj2.to(device)
    masked_model.mask_logits_down.data = mask_logits_down.to(device)

    # Disable dropout
    masked_model.base_model.dropout.p = 0.0
    for layer in masked_model.base_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    masked_model.eval()

    with torch.no_grad():
        # Teacher predictions
        teacher_logits = teacher_model(data, attention_mask)[0]
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Masked model predictions
        student_logits = masked_model(data, attention_mask)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL divergence
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean').item()

        # CE losses
        targets = data[:, 1:]
        target_mask = attention_mask[:, 1:]

        teacher_ce_loss_raw = F.cross_entropy(
            teacher_logits[:, :-1, :].reshape(-1, teacher_logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )
        teacher_ce = (teacher_ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
        teacher_ce = teacher_ce.item()

        student_ce_loss_raw = F.cross_entropy(
            student_logits[:, :-1, :].reshape(-1, student_logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )
        student_ce = (student_ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
        student_ce = student_ce.item()

        ce_diff = student_ce - teacher_ce

        # Sparsity
        mask_proj1, mask_proj2, mask_down = masked_model.get_masks()
        total_params = mask_proj1.numel() + mask_proj2.numel() + mask_down.numel()
        active_params = mask_proj1.sum().item() + mask_proj2.sum().item() + mask_down.sum().item()
        sparsity = 1.0 - (active_params / total_params)

    return {
        'kl_div': kl_div,
        'teacher_ce': teacher_ce,
        'student_ce': student_ce,
        'ce_diff': ce_diff,
        'abs_ce_diff': abs(ce_diff),
        'sparsity': sparsity,
        'active_params': int(active_params),
        'total_params': int(total_params)
    }
# %%
# Compute gradients for each sample

print("\n" + "="*80)
print("Computing gradients with mask logits initialized to 0")
print("="*80)

all_gradients = []

for sample_idx, (data, attention_mask) in enumerate(samples):
    print(f"\nSample {sample_idx}:")

    grads = compute_initial_gradients(teacher_model, data, attention_mask, device=DEVICE, init_value=0.5)
    all_gradients.append(grads)

    # Print statistics
    print(f"  KL loss (all weights masked): {grads['kl_loss']:.6f}")
    print(f"  Gradient norms:")
    print(f"    proj1: {grads['grad_proj1'].norm().item():.6f}")
    print(f"    proj2: {grads['grad_proj2'].norm().item():.6f}")
    print(f"    down:  {grads['grad_down'].norm().item():.6f}")
    print(f"  Gradient statistics (proj1):")
    print(f"    mean: {grads['grad_proj1'].mean().item():.6f}")
    print(f"    std:  {grads['grad_proj1'].std().item():.6f}")
    print(f"    min:  {grads['grad_proj1'].min().item():.6f}")
    print(f"    max:  {grads['grad_proj1'].max().item():.6f}")

# %%
# Visualize gradient distributions

fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))

for sample_idx in range(n_samples):
    grads = all_gradients[sample_idx]

    # Plot proj1 gradients
    ax = axes[sample_idx, 0] if n_samples > 1 else axes[0]
    grad_flat = grads['grad_proj1'].cpu().flatten().numpy()
    ax.hist(grad_flat, bins=100, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Gradient value')
    ax.set_ylabel('Count')
    ax.set_title(f'Sample {sample_idx}: proj1 gradients')
    ax.grid(True, alpha=0.3)

    # Plot proj2 gradients
    ax = axes[sample_idx, 1] if n_samples > 1 else axes[1]
    grad_flat = grads['grad_proj2'].cpu().flatten().numpy()
    ax.hist(grad_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Gradient value')
    ax.set_ylabel('Count')
    ax.set_title(f'Sample {sample_idx}: proj2 gradients')
    ax.grid(True, alpha=0.3)

    # Plot down gradients
    ax = axes[sample_idx, 2] if n_samples > 1 else axes[2]
    grad_flat = grads['grad_down'].cpu().flatten().numpy()
    ax.hist(grad_flat, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Gradient value')
    ax.set_ylabel('Count')
    ax.set_title(f'Sample {sample_idx}: down gradients')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = FIGURES_DIR / 'mask_gradient_distributions_init0.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved gradient distributions to '{save_path}'")
plt.show()

# %%
grad_proj1 = grads["grad_proj1"]
w_proj1 = teacher_model.layers[0].bilinear.proj1.weight
# Check how many have the same sign
small_grads_amount = (grad_proj1 < 1e-10).sum().item()
large_grads_mask = (grad_proj1 > 1e-10)
print(f"Number of small gradients: {small_grads_amount}")
print(f"Percentage of small gradients: {small_grads_amount / grad_proj1.numel():.2%}")
same_sign = (grad_proj1 * w_proj1 > 0).float()
print(f"Percentage of weights with the same sign: {same_sign.mean().item():.2%}")

# find which large grads have the same sign
large_grads_mask = (grad_proj1 > 1e-10)
large_grads_same_sign = (grad_proj1[large_grads_mask] * w_proj1[large_grads_mask] > 0).float()
print(f"Percentage of large gradients with the same sign: {large_grads_same_sign.mean().item():.2%}")

# plot a histogram of the grads
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(grad_proj1.cpu().flatten().numpy(), bins=100, alpha=0.7, color='red', edgecolor='black')
ax.set_xlabel('Gradient value')
ax.set_ylabel('Count')
ax.set_title('Gradients of proj1 weights')
ax.grid(True, alpha=0.3)
plt.show()

# %%
# 1st-order and 2nd-order importance scores
scores_first, scores_second = [], []
params = []
for w, g in zip(w_proj1, grad_proj1):
    s1 = (g * g).abs()
    s2 = 0.5 * (w**2) * (g**2)
    scores_first.append(s1.view(-1))
    scores_second.append(s2.view(-1))

s1 = torch.cat(scores_first)
s2 = torch.cat(scores_second)

# Option A: keep top-k by 2nd-order score
k = int(0.05 * s2.numel())  # keep 5% most important
thresh = s2.kthvalue(s2.numel() - k).values
keep_flat = (s2 >= thresh)

# %%
# Second-order importance scores using finite differences (Hessian approximation)

def compute_kl_gradient_wrt_param(param, teacher_model, data, attention_mask, device='cuda'):
    """
    Compute KL divergence gradient w.r.t. a single parameter tensor.

    Args:
        param: The parameter tensor to compute gradient for (e.g., bilinear.proj1.weight)
        teacher_model: Teacher model
        data, attention_mask: Input data
        device: Device

    Returns:
        Gradient tensor same shape as param
    """
    # Enable gradient for this parameter
    param.requires_grad_(True)
    if param.grad is not None:
        param.grad.zero_()

    # Get teacher predictions
    with torch.no_grad():
        teacher_logits = teacher_model(data, attention_mask)[0]
        teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Forward pass
    student_logits = teacher_model(data, attention_mask)[0]
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # KL divergence
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    # Backward
    kl_loss.backward()

    grad = param.grad.clone()
    param.requires_grad_(False)

    return grad


def compute_second_order_scores(param, teacher_model, data, attention_mask,
                                eps=1e-3, device='cuda'):
    """
    Compute second-order importance scores using finite difference Hessian approximation.

    Score = 0.5 * Delta * H * Delta, where:
    - Delta = eps * W (perturbation)
    - H is approximated via finite differences: (g_plus - g_minus) / (2 * Delta)

    Args:
        param: Weight parameter to analyze (e.g., bilinear.proj1.weight)
        teacher_model: Teacher model
        data, attention_mask: Input data
        eps: Relative perturbation size (default 1e-3 = 0.1%)
        device: Device

    Returns:
        scores: Second-order importance scores (same shape as param)
    """
    # Save original weights
    W_orig = param.detach().clone()

    # Compute gradient at +eps perturbation
    with torch.no_grad():
        param.mul_(1 + eps)
    g_plus = compute_kl_gradient_wrt_param(param, teacher_model, data, attention_mask, device)

    # Compute gradient at -eps perturbation
    with torch.no_grad():
        param.copy_(W_orig * (1 - eps))
    g_minus = compute_kl_gradient_wrt_param(param, teacher_model, data, attention_mask, device)

    # Restore original weights
    with torch.no_grad():
        param.copy_(W_orig)

    # Estimate Hessian-vector product: H * Delta ≈ (g_plus - g_minus) / (2 * Delta)
    Delta = eps * W_orig
    HDelta = (g_plus - g_minus) / 2.0  # Already divided by Delta in the formula

    # Second-order score: 0.5 * Delta^T * H * Delta
    score = 0.5 * Delta * HDelta
    score = score.clamp_min_(0)  # Clamp negative values for numerical stability

    return score

def compute_iterated_second_order_scores(param, teacher_model, data, attention_mask,
                                        n_iterations=3, sparsity_per_iter=0.5,
                                        eps=1e-3, device='cuda'):
    """
    Iteratively prune weights using second-order importance scores.

    Algorithm:
    1. Compute 2nd-order scores on current weights
    2. Prune bottom x% by score
    3. Update weights (zero out pruned weights)
    4. Repeat

    Args:
        param: Weight parameter to prune (will be temporarily modified)
        teacher_model: Teacher model
        data, attention_mask: Input data
        n_iterations: Number of pruning iterations (default 3)
        sparsity_per_iter: Fraction to prune each iteration (default 0.5 = 50%)
        eps: Perturbation size for finite differences
        device: Device

    Returns:
        final_mask: Binary mask indicating which weights survived (1=keep, 0=pruned)
        all_scores: List of score tensors from each iteration
        cumulative_mask: Cumulative mask showing what got pruned when
    """
    # Save original weights
    W_orig = param.detach().clone()

    all_scores = []
    cumulative_mask = torch.ones_like(param)  # Start with all weights active

    for iteration in range(n_iterations):
        print(f"  Iteration {iteration + 1}/{n_iterations}:")

        # Compute second-order scores on current (possibly pruned) weights
        scores = compute_second_order_scores(param, teacher_model, data, attention_mask, eps, device)
        all_scores.append(scores.detach().clone())

        # Only consider scores for weights that are still active
        masked_scores = scores * cumulative_mask

        # Determine how many weights to keep this iteration
        n_active = cumulative_mask.sum().item()
        n_to_keep = int(n_active * (1 - sparsity_per_iter))

        print(f"    Active weights: {int(n_active)}")
        print(f"    Keeping: {n_to_keep}")
        print(f"    Pruning: {int(n_active) - n_to_keep}")

        if n_to_keep <= 0:
            print(f"    Warning: Would prune all remaining weights, stopping early")
            break

        # Find threshold for top-k scores among active weights
        active_scores = masked_scores[cumulative_mask > 0]
        if len(active_scores) > n_to_keep:
            threshold = torch.kthvalue(active_scores, len(active_scores) - n_to_keep).values
        else:
            threshold = active_scores.min() - 1  # Keep everything

        # Create iteration mask: keep weights with score >= threshold
        iter_mask = (masked_scores >= threshold).float()

        # Update cumulative mask
        cumulative_mask = cumulative_mask * iter_mask

        # Apply mask to actual parameter
        with torch.no_grad():
            param.mul_(iter_mask)

        current_sparsity = 1.0 - cumulative_mask.sum().item() / cumulative_mask.numel()
        print(f"    Current sparsity: {current_sparsity:.2%}")

    # Restore original weights (don't leave the model modified)
    with torch.no_grad():
        param.copy_(W_orig)

    final_mask = cumulative_mask

    return final_mask, all_scores, cumulative_mask
sample_idx = 0
data, attention_mask = samples[sample_idx]

print("\n" + "="*80)
print("Computing second-order importance scores (Hessian approximation)")
print("="*80)

# Get the parameter
param_proj1 = teacher_model.layers[0].bilinear.proj1.weight

print(f"\nComputing scores for proj1 weights (shape: {param_proj1.shape})")
print("This may take a moment (requires 2 forward+backward passes)...")

second_order_scores = compute_second_order_scores(
    param_proj1, teacher_model, data, attention_mask,
    eps=1e-3, device=DEVICE
)

print(f"\nSecond-order score statistics:")
print(f"  Mean: {second_order_scores.mean().item():.6e}")
print(f"  Std:  {second_order_scores.std().item():.6e}")
print(f"  Min:  {second_order_scores.min().item():.6e}")
print(f"  Max:  {second_order_scores.max().item():.6e}")

# Compare with first-order scores (gradient magnitude)
first_order_scores = grads['grad_proj1'].abs()

print(f"\nFirst-order score statistics (gradient magnitude):")
print(f"  Mean: {first_order_scores.mean().item():.6e}")
print(f"  Std:  {first_order_scores.std().item():.6e}")
print(f"  Min:  {first_order_scores.min().item():.6e}")
print(f"  Max:  {first_order_scores.max().item():.6e}")

print("\n" + "="*80)
print("Computing ITERATED second-order importance scores")
print("="*80)

final_mask, all_scores_iter, cumulative_mask = compute_iterated_second_order_scores(
    param_proj1, teacher_model, data, attention_mask,
    n_iterations=3,
    sparsity_per_iter=0.5,  # Prune 50% of remaining weights each iteration
    eps=1e-3,
    device=DEVICE
)

print(f"\nFinal sparsity: {1.0 - final_mask.sum().item() / final_mask.numel():.2%}")
print(f"Weights remaining: {int(final_mask.sum().item())} / {final_mask.numel()}")

for i in range(len(all_scores_iter)):
    print(f"\nIteration {i+1} score statistics:")
    print(f"  Mean: {all_scores_iter[i].mean().item():.6e}")
    print(f"  Std:  {all_scores_iter[i].std().item():.6e}")
    print(f"  Min:  {all_scores_iter[i].min().item():.6e}")
    print(f"  Max:  {all_scores_iter[i].max().item():.6e}")

# %%
# Create masks based on second-order scores and evaluate

print("\n" + "="*80)
print("Creating masks from second-order scores and comparing methods")
print("="*80)

# Lists to store results for plotting
CE_diffs_1st_order = []
CE_diffs_2nd_order = []
CE_diffs_iterated = []
KL_diffs_1st_order = []
KL_diffs_2nd_order = []
KL_diffs_iterated = []

# Sweep over sparsity values (5% to 100% in 5% increments)
print("\nSweeping sparsity from 5% to 100% in 5% increments...")
for sparsity_target in [i/20 for i in range(1, 21)]:
    k = int((1 - sparsity_target) * second_order_scores.numel())
    thresh = torch.kthvalue(second_order_scores.view(-1), second_order_scores.numel() - k).values

    print(f"\nTarget sparsity: {sparsity_target:.1%}")
    print(f"Keeping top {k} weights out of {second_order_scores.numel()}")
    print(f"Threshold: {thresh.item():.6e}")

    # Create mask logits based on second-order scores
    mask_logits_proj1_2nd = torch.where(
        second_order_scores >= thresh,
        torch.tensor(10.0),
        torch.tensor(-10.0)
    )

    # For other layers, keep all active (for this example)
    mask_logits_proj2_2nd = torch.ones_like(teacher_model.layers[0].bilinear.proj2.weight)
    mask_logits_down_2nd = torch.ones_like(teacher_model.layers[0].bilinear.down.weight)

    # Evaluate this configuration
    config_2nd_order = evaluate_mask_configuration(
        teacher_model,
        mask_logits_proj1_2nd,
        mask_logits_proj2_2nd,
        mask_logits_down_2nd,
        data, attention_mask, device=DEVICE
    )

    print(f"\nSecond-order mask (50% sparsity on proj1 only):")
    for key, val in config_2nd_order.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

    # Compare with first-order (gradient magnitude) mask
    thresh_1st = torch.kthvalue(first_order_scores.view(-1), first_order_scores.numel() - k).values
    mask_logits_proj1_1st = torch.where(
        first_order_scores >= thresh_1st,
        torch.tensor(10.0),
        torch.tensor(-10.0)
    )
    # mask_logits_proj1_1st = torch.ones_like(teacher_model.layers[0].bilinear.proj1.weight)

    config_1st_order = evaluate_mask_configuration(
        teacher_model,
        mask_logits_proj1_1st,
        mask_logits_proj2_2nd,  # Same as above
        mask_logits_down_2nd,   # Same as above
        data, attention_mask, device=DEVICE
    )

    print(f"\nFirst-order mask (50% sparsity on proj1 only):")
    for key, val in config_1st_order.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

    # Compute iterated pruning for this sparsity level
    # Need to determine how many iterations to get to target sparsity
    # If we prune 50% each iteration: 0.5, 0.75, 0.875, 0.9375, ...
    # Solve: 1 - (1 - sparsity_per_iter)^n = sparsity_target
    # n = log(1 - sparsity_target) / log(1 - sparsity_per_iter)
    sparsity_per_iter = 0.5
    if sparsity_target < 0.99:  # Avoid log(0)
        n_iter = max(1, int(np.ceil(np.log(1 - sparsity_target) / np.log(1 - sparsity_per_iter))))
    else:
        n_iter = 5  # For very high sparsity

    # Compute iterated mask
    mask_iterated, _, _ = compute_iterated_second_order_scores(
        param_proj1, teacher_model, data, attention_mask,
        n_iterations=n_iter,
        sparsity_per_iter=sparsity_per_iter,
        eps=1e-3,
        device=DEVICE
    )

    # Convert to mask logits
    mask_logits_proj1_iter = torch.where(
        mask_iterated > 0.5,
        torch.tensor(10.0),
        torch.tensor(-10.0)
    )

    config_iterated = evaluate_mask_configuration(
        teacher_model,
        mask_logits_proj1_iter,
        mask_logits_proj2_2nd,
        mask_logits_down_2nd,
        data, attention_mask, device=DEVICE
    )

    print(f"\nIterated pruning ({n_iter} iterations):")
    for key, val in config_iterated.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

    print(f"\nComparison:")
    print(f"  First-order KL:  {config_1st_order['kl_div']:.6f}")
    print(f"  Second-order KL: {config_2nd_order['kl_div']:.6f}")
    print(f"  Iterated KL:     {config_iterated['kl_div']:.6f}")

    # Store results
    CE_diffs_1st_order.append(config_1st_order['ce_diff'])
    CE_diffs_2nd_order.append(config_2nd_order['ce_diff'])
    CE_diffs_iterated.append(config_iterated['ce_diff'])
    KL_diffs_1st_order.append(config_1st_order['kl_div'])
    KL_diffs_2nd_order.append(config_2nd_order['kl_div'])
    KL_diffs_iterated.append(config_iterated['kl_div'])

# Plot CE and KL differences
print("\n" + "="*80)
print("Plotting comparison of pruning methods")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sparsity_labels = [f"{i/20:.1%}" for i in range(1, 21)]

# Plot 1: CE differences
axes[0].plot(sparsity_labels, CE_diffs_1st_order, marker='o', label='1st-order (gradient)', linewidth=2)
axes[0].plot(sparsity_labels, CE_diffs_2nd_order, marker='s', label='2nd-order (one-shot)', linewidth=2)
axes[0].plot(sparsity_labels, CE_diffs_iterated, marker='^', label='2nd-order (iterated)', linewidth=2)
axes[0].set_xlabel('Sparsity Target', fontsize=12)
axes[0].set_ylabel('CE Difference (+ means worse)', fontsize=12)
axes[0].set_title('Cross-Entropy Difference vs Sparsity', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(len(sparsity_labels)))
axes[0].set_xticklabels(sparsity_labels, rotation=45)
axes[0].set_yscale('log')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: KL divergences
axes[1].plot(sparsity_labels, KL_diffs_1st_order, marker='o', label='1st-order (gradient)', linewidth=2)
axes[1].plot(sparsity_labels, KL_diffs_2nd_order, marker='s', label='2nd-order (one-shot)', linewidth=2)
axes[1].plot(sparsity_labels, KL_diffs_iterated, marker='^', label='2nd-order (iterated)', linewidth=2)
axes[1].set_xlabel('Sparsity Target', fontsize=12)
axes[1].set_ylabel('KL Divergence', fontsize=12)
axes[1].set_title('KL Divergence vs Sparsity', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(sparsity_labels)))
axes[1].set_xticklabels(sparsity_labels, rotation=45)
axes[1].set_yscale('log')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_path = FIGURES_DIR / 'pruning_methods_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved comparison plot to '{save_path}'")
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("Summary: Which method performs best?")
print("="*80)

for i, sparsity in enumerate([i/20 for i in range(1, 21)]):
    if i % 4 == 0:  # Print every 4th sparsity level (20%, 40%, 60%, 80%, 100%)
        print(f"\nSparsity {sparsity:.0%}:")
        print(f"  1st-order KL: {KL_diffs_1st_order[i]:.6f}")
        print(f"  2nd-order KL: {KL_diffs_2nd_order[i]:.6f}")
        print(f"  Iterated KL:  {KL_diffs_iterated[i]:.6f}")
        best_method = min(
            ('1st-order', KL_diffs_1st_order[i]),
            ('2nd-order', KL_diffs_2nd_order[i]),
            ('Iterated', KL_diffs_iterated[i]),
            key=lambda x: x[1]
        )
        print(f"  Best: {best_method[0]} (KL={best_method[1]:.6f})")
# %%
# Optional: Load trained masks and compare

def load_trained_masks(target_sparsity, sample_idx, mask_dir=None):
    """Load trained mask weights"""
    if mask_dir is None:
        mask_dir = MASK_WEIGHTS_DIR
    filename = Path(mask_dir) / f'mask_weights_lambda{target_sparsity:.0e}_sample{sample_idx}.pt'
    if not filename.exists():
        print(f"Warning: Mask file not found: {filename}")
        return None

    mask_state = torch.load(str(filename), map_location='cpu')
    return mask_state

# Try to load a trained mask (e.g., lambda=100.0, sample=0)
target_lambda = 100.0
sample_idx_to_load = 0

trained_masks = load_trained_masks(target_lambda, sample_idx_to_load)

if trained_masks is not None:
    print(f"\n{'='*80}")
    print(f"Loaded trained masks: lambda={target_lambda:.0e}, sample={sample_idx_to_load}")
    print(f"{'='*80}")

    # Print mask statistics
    for mask_name in ['mask_logits_proj1', 'mask_logits_proj2', 'mask_logits_down']:
        mask_logits = trained_masks[mask_name]
        binary_mask = (mask_logits > 0).float()
        sparsity = 1.0 - binary_mask.mean().item()

        print(f"\n{mask_name}:")
        print(f"  Shape: {mask_logits.shape}")
        print(f"  Sparsity (logits > 0): {sparsity:.2%}")
        print(f"  Logits - mean: {mask_logits.mean().item():.4f}, std: {mask_logits.std().item():.4f}")
        print(f"  Logits - min: {mask_logits.min().item():.4f}, max: {mask_logits.max().item():.4f}")

# %%
# Compare gradient at init=0 vs trained mask values

if trained_masks is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Get gradients for this sample
    grads = all_gradients[sample_idx_to_load]

    mask_names = ['proj1', 'proj2', 'down']

    for idx, mask_name in enumerate(mask_names):
        ax = axes[idx]

        # Get gradient and trained mask
        grad = grads[f'grad_{mask_name}'].cpu().flatten().numpy()
        trained_logits = trained_masks[f'mask_logits_{mask_name}'].flatten().numpy()

        # Create 2D histogram
        h = ax.hist2d(grad, trained_logits, bins=50, cmap='viridis', cmin=1)
        ax.set_xlabel('Initial gradient (logits=0)', fontsize=11)
        ax.set_ylabel('Final trained logit value', fontsize=11)
        ax.set_title(f'{mask_name}: Gradient vs Trained Value', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Decision boundary')
        ax.legend()
        plt.colorbar(h[3], ax=ax, label='Count')

    plt.tight_layout()
    save_path = FIGURES_DIR / 'gradient_vs_trained_masks.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to '{save_path}'")
    plt.show()

# %%
# Create scaled-down teacher model (bilinear weights * 0.1)

def create_scaled_teacher(teacher_model, scale_factor=0.1, layer_idx=0):
    """
    Create a copy of teacher model with bilinear weights scaled down.

    Args:
        teacher_model: Original teacher model
        scale_factor: Factor to scale bilinear weights (default 0.1 = 1/10)
        layer_idx: Which layer to scale (default 0)

    Returns:
        Scaled model (detached copy)
    """
    # Create a deep copy of the model
    import copy
    scaled_model = copy.deepcopy(teacher_model)

    # Scale the bilinear weights
    bilinear = scaled_model.layers[layer_idx].bilinear
    bilinear.proj1.weight.data *= scale_factor
    bilinear.proj2.weight.data *= scale_factor
    bilinear.down.weight.data *= scale_factor

    # Also scale biases if desired
    bilinear.proj1.bias.data *= scale_factor
    bilinear.proj2.bias.data *= scale_factor
    bilinear.down.bias.data *= scale_factor

    scaled_model.eval()
    return scaled_model

print("Creating scaled teacher model (bilinear weights * 0.1)...")
scaled_teacher = create_scaled_teacher(teacher_model, scale_factor=0.1, layer_idx=0)

# Verify the scaling
orig_proj1_norm = teacher_model.layers[0].bilinear.proj1.weight.norm().item()
scaled_proj1_norm = scaled_teacher.layers[0].bilinear.proj1.weight.norm().item()
print(f"Original proj1 weight norm: {orig_proj1_norm:.4f}")
print(f"Scaled proj1 weight norm:   {scaled_proj1_norm:.4f}")
print(f"Ratio: {scaled_proj1_norm / orig_proj1_norm:.4f}")

# %%
# Compute gradients to match scaled teacher (instead of all-zeros)

def compute_gradients_to_scaled_teacher(original_teacher, scaled_teacher, data, attention_mask,
                                       init_value=0.0, device='cuda'):
    """
    Compute gradients for mask logits when trying to match a scaled-down teacher.
    The masked model starts with logits=init_value and tries to minimize KL divergence
    between its output and the scaled teacher's output.

    Returns gradients and metrics.
    """
    # Create masked model with specified initialization
    masked_model = MaskedTransformer(original_teacher, mask_layer_idx=0, init_value=init_value).to(device)

    # Disable dropout
    masked_model.base_model.dropout.p = 0.0
    for layer in masked_model.base_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    masked_model.train()

    # Get scaled teacher predictions (TARGET)
    with torch.no_grad():
        scaled_teacher_logits = scaled_teacher(data, attention_mask)[0]
        scaled_teacher_probs = F.softmax(scaled_teacher_logits, dim=-1)

        # Also get original teacher for comparison
        orig_teacher_logits = original_teacher(data, attention_mask)[0]
        orig_teacher_probs = F.softmax(orig_teacher_logits, dim=-1)

        # KL divergence between scaled and original teacher
        teacher_kl = F.kl_div(
            F.log_softmax(scaled_teacher_logits, dim=-1),
            orig_teacher_probs,
            reduction='batchmean'
        ).item()

        # CE for both teachers
        targets = data[:, 1:]
        target_mask = attention_mask[:, 1:]

        scaled_ce_loss_raw = F.cross_entropy(
            scaled_teacher_logits[:, :-1, :].reshape(-1, scaled_teacher_logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )
        scaled_teacher_ce = (scaled_ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
        scaled_teacher_ce = scaled_teacher_ce.item()

        orig_ce_loss_raw = F.cross_entropy(
            orig_teacher_logits[:, :-1, :].reshape(-1, orig_teacher_logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )
        orig_teacher_ce = (orig_ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
        orig_teacher_ce = orig_teacher_ce.item()

    # Forward pass with masked model
    student_logits = masked_model(data, attention_mask)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # KL divergence loss (student -> scaled teacher)
    kl_loss = F.kl_div(student_log_probs, scaled_teacher_probs, reduction='batchmean')

    # Compute CE for masked model
    ce_loss_raw = F.cross_entropy(
        student_logits[:, :-1, :].reshape(-1, student_logits.size(-1)),
        targets.reshape(-1),
        reduction='none'
    )
    masked_ce = (ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
    masked_ce = masked_ce.item()

    # Backward pass
    kl_loss.backward()

    # Extract gradients
    grad_proj1 = masked_model.mask_logits_proj1.grad.clone()
    grad_proj2 = masked_model.mask_logits_proj2.grad.clone()
    grad_down = masked_model.mask_logits_down.grad.clone()

    return {
        'grad_proj1': grad_proj1,
        'grad_proj2': grad_proj2,
        'grad_down': grad_down,
        'kl_loss': kl_loss.item(),
        'teacher_kl': teacher_kl,  # KL between scaled and original teacher
        'scaled_teacher_ce': scaled_teacher_ce,
        'orig_teacher_ce': orig_teacher_ce,
        'masked_ce': masked_ce,
        'ce_diff_vs_scaled': masked_ce - scaled_teacher_ce,
        'ce_diff_vs_orig': masked_ce - orig_teacher_ce,
    }

# %%
# Compute gradients for matching scaled teacher

print("\n" + "="*80)
print("Computing gradients to match SCALED teacher (bilinear * 0.1)")
print("="*80)

scaled_gradients = []

for sample_idx, (data, attention_mask) in enumerate(samples):
    print(f"\nSample {sample_idx}:")

    grads = compute_gradients_to_scaled_teacher(
        teacher_model, scaled_teacher, data, attention_mask,
        init_value=0.0, device=DEVICE
    )
    scaled_gradients.append(grads)

    # Print statistics
    print(f"  KL (scaled teacher vs original): {grads['teacher_kl']:.6f}")
    print(f"  KL (masked model vs scaled teacher): {grads['kl_loss']:.6f}")
    print(f"  CE diff (masked vs scaled): {grads['ce_diff_vs_scaled']:+.6f}")
    print(f"  CE diff (masked vs original): {grads['ce_diff_vs_orig']:+.6f}")
    print(f"  Gradient norms:")
    print(f"    proj1: {grads['grad_proj1'].norm().item():.6f}")
    print(f"    proj2: {grads['grad_proj2'].norm().item():.6f}")
    print(f"    down:  {grads['grad_down'].norm().item():.6f}")

# %%
# Compare gradients: all-zeros vs scaled teacher

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

sample_to_plot = 0  # Plot first sample
grads_zeros = all_gradients[sample_to_plot]
grads_scaled = scaled_gradients[sample_to_plot]

mask_names = ['proj1', 'proj2', 'down']

for row_idx, mask_name in enumerate(mask_names):
    # Column 1: Gradients when matching all-zeros (original analysis)
    ax = axes[row_idx, 0]
    grad_flat = grads_zeros[f'grad_{mask_name}'].cpu().flatten().numpy()
    ax.hist(grad_flat, bins=100, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Gradient value')
    ax.set_ylabel('Count')
    ax.set_title(f'{mask_name}: Gradients (init=0, target=all zeros)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Column 2: Gradients when matching scaled teacher
    ax = axes[row_idx, 1]
    grad_flat = grads_scaled[f'grad_{mask_name}'].cpu().flatten().numpy()
    ax.hist(grad_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Gradient value')
    ax.set_ylabel('Count')
    ax.set_title(f'{mask_name}: Gradients (init=0, target=scaled teacher)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Column 3: Scatter plot comparing the two
    ax = axes[row_idx, 2]
    grad_zeros_flat = grads_zeros[f'grad_{mask_name}'].cpu().flatten().numpy()
    grad_scaled_flat = grads_scaled[f'grad_{mask_name}'].cpu().flatten().numpy()

    # Subsample for plotting if too many points
    if len(grad_zeros_flat) > 10000:
        indices = np.random.choice(len(grad_zeros_flat), 10000, replace=False)
        grad_zeros_flat = grad_zeros_flat[indices]
        grad_scaled_flat = grad_scaled_flat[indices]

    ax.scatter(grad_zeros_flat, grad_scaled_flat, alpha=0.3, s=1)
    ax.set_xlabel('Gradient (target=zeros)')
    ax.set_ylabel('Gradient (target=scaled)')
    ax.set_title(f'{mask_name}: Gradient comparison', fontweight='bold')
    ax.plot([grad_zeros_flat.min(), grad_zeros_flat.max()],
            [grad_zeros_flat.min(), grad_zeros_flat.max()],
            'r--', alpha=0.5, label='y=x')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = FIGURES_DIR / 'gradient_comparison_zeros_vs_scaled.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved comparison to '{save_path}'")
plt.show()

# %%
# Compare scaled-teacher gradients with trained masks

if trained_masks is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    grads = scaled_gradients[sample_idx_to_load]

    for idx, mask_name in enumerate(mask_names):
        ax = axes[idx]

        # Get gradient and trained mask
        grad = grads[f'grad_{mask_name}'].cpu().flatten().numpy()
        trained_logits = trained_masks[f'mask_logits_{mask_name}'].flatten().numpy()

        # Create 2D histogram
        h = ax.hist2d(grad, trained_logits, bins=50, cmap='viridis', cmin=1)
        ax.set_xlabel('Gradient (target=scaled teacher)', fontsize=11)
        ax.set_ylabel('Final trained logit value', fontsize=11)
        ax.set_title(f'{mask_name}: Scaled-Teacher Grad vs Trained Value',
                    fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label='Decision boundary')
        ax.legend()
        plt.colorbar(h[3], ax=ax, label='Count')

    plt.tight_layout()
    save_path = FIGURES_DIR / 'scaled_gradient_vs_trained_masks.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to '{save_path}'")
    plt.show()



# %%
# Example: Evaluate different mask configurations

print("\n" + "="*80)
print("Example: Evaluating different mask configurations")
print("="*80)

# Choose a sample to evaluate
eval_sample_idx = 0
eval_data, eval_attention_mask = samples[eval_sample_idx]

# Configuration 1: All weights active (logits = 1.0)
print("\n1. All weights active (logits=1.0):")
config1 = evaluate_mask_configuration(
    teacher_model,
    torch.ones_like(teacher_model.layers[0].bilinear.proj1.weight) * 1.0,
    torch.ones_like(teacher_model.layers[0].bilinear.proj2.weight) * 1.0,
    torch.ones_like(teacher_model.layers[0].bilinear.down.weight) * 1.0,
    eval_data, eval_attention_mask, device=DEVICE
)
for key, val in config1.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.6f}")
    else:
        print(f"  {key}: {val}")

# Configuration 2: All weights masked (logits = -10.0)
print("\n2. All weights masked (logits=-10.0):")
config2 = evaluate_mask_configuration(
    teacher_model,
    torch.ones_like(teacher_model.layers[0].bilinear.proj1.weight) * -10.0,
    torch.ones_like(teacher_model.layers[0].bilinear.proj2.weight) * -10.0,
    torch.ones_like(teacher_model.layers[0].bilinear.down.weight) * -10.0,
    eval_data, eval_attention_mask, device=DEVICE
)
for key, val in config2.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.6f}")
    else:
        print(f"  {key}: {val}")

# Configuration 3: Trained masks (if available)
if trained_masks is not None and sample_idx_to_load == eval_sample_idx:
    print(f"\n3. Trained masks (lambda={target_lambda:.0e}):")
    config3 = evaluate_mask_configuration(
        teacher_model,
        trained_masks['mask_logits_proj1'],
        trained_masks['mask_logits_proj2'],
        trained_masks['mask_logits_down'],
        eval_data, eval_attention_mask, device=DEVICE
    )
    for key, val in config3.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

# Configuration 4: Top-k by gradient magnitude (scaled teacher gradients)
print("\n4. Top-k weights by gradient magnitude (k=10% of params, using scaled-teacher gradients):")
grads_for_topk = scaled_gradients[eval_sample_idx]

# Flatten all gradients
all_grads = torch.cat([
    grads_for_topk['grad_proj1'].flatten(),
    grads_for_topk['grad_proj2'].flatten(),
    grads_for_topk['grad_down'].flatten()
])

# Get top 10% by absolute gradient
k = int(0.1 * len(all_grads))
topk_threshold = torch.topk(all_grads.abs(), k).values[-1]

# Create masks based on threshold
mask_proj1_topk = torch.where(
    grads_for_topk['grad_proj1'].abs() >= topk_threshold,
    torch.tensor(10.0),
    torch.tensor(-10.0)
)
mask_proj2_topk = torch.where(
    grads_for_topk['grad_proj2'].abs() >= topk_threshold,
    torch.tensor(10.0),
    torch.tensor(-10.0)
)
mask_down_topk = torch.where(
    grads_for_topk['grad_down'].abs() >= topk_threshold,
    torch.tensor(10.0),
    torch.tensor(-10.0)
)

config4 = evaluate_mask_configuration(
    teacher_model,
    mask_proj1_topk,
    mask_proj2_topk,
    mask_down_topk,
    eval_data, eval_attention_mask, device=DEVICE
)
for key, val in config4.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.6f}")
    else:
        print(f"  {key}: {val}")

# %%
print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
print("\nTo evaluate a custom mask configuration, use:")
print("  evaluate_mask_configuration(teacher_model, mask_logits_proj1,")
print("                              mask_logits_proj2, mask_logits_down,")
print("                              data, attention_mask, device=DEVICE)")
print("\nAvailable samples: samples[0] to samples[{}]".format(len(samples)-1))
print("Available gradients: all_gradients[i] (init=0) and scaled_gradients[i] (scaled teacher)")

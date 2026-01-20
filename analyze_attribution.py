# %%
"""
Attribution Analysis Script

Interactive analysis of frozen-RMS trained model.
Run cells individually to explore attributions and ablations.

=============================================================================
FUNCTION REFERENCE TABLE
=============================================================================

CORE FUNCTIONS (attribution & ablation):
  get_attribution(x, y)                    - Get frozen-RMS attribution for one image
  get_all_logits(x)                        - Get model predictions for all classes
  ablate_channel(x, y, layer, patch, ch)   - Zero one channel, get new logit
  ablate_multiple_channels(x, y, list)     - Zero multiple channels at once
  get_top_channels(all_attr, k=20)         - Find top-k channels by |attribution|
  verify_attribution(x, y, top_k=10)       - Compare attribution vs actual ablation effect

VISUALIZATION:
  show_image(x)                            - Display CIFAR image (undo normalization)
  plot_attribution_heatmap(all_attr, layer)- 8x8 heatmap of attribution per patch
  plot_channel_histogram(all_attr, layer)  - Histogram of attribution values

DISTRIBUTION PLOTS:
  plot_channel_distributions(all_attr, all_h, channels=None, n_channels=20, n_per_plot=5)
                                           - Histograms of attr & h, 5 per subplot
  plot_attribution_vs_activation_scatter() - Scatter: attr vs h for top channels

FIRING MASKS:
  compute_firing_masks(all_h, method)      - Binary masks: 'threshold'/'percentile'/'sign'/'significant'
  compute_firing_rate(masks)               - Fraction of patches where channel fires

CO-FIRING CORRELATION:
  compute_cofire_correlation(all_h, method)- Channel correlation: 'continuous'/'binary'/'jaccard'
  compute_signed_cofire(all_h)             - Same-sign vs opposite-sign co-firing
  plot_cofire_matrix(corr)                 - Visualize correlation matrix

CAUSAL CONNECTIVITY (from weights, no data needed):
  compute_causal_connectivity()            - Get D norms, channel→class effects from weights
  plot_channel_class_effects(connectivity) - Heatmap of channel effects on classes
  plot_top_channels_per_class(connectivity)- Bar charts: top promote/suppress per class
  print_top_channels_per_class(connectivity)- Text printout of above

SPATIAL PATTERNS (where in the 8x8 patch grid a channel fires):
  analyze_spatial_patterns(all_h)          - Stats: variance, x/y correlation, locality
  plot_channel_spatial_pattern(all_h, ch)  - 8x8 heatmap of one channel's spatial pattern
  plot_multiple_spatial_patterns(all_h)    - Grid of spatial patterns for many channels

CLUSTERING:
  cluster_channels(all_attr, all_h, n_clusters, feature)
                                           - KMeans clustering by 'activation'/'attribution'/'combined'
  analyze_clusters(labels, connectivity)   - Print cluster→class associations

FEATURE GROUPS BY CLASS:
  find_feature_groups_by_class(n_samples)  - Which channels matter for each class (uses data)
  plot_class_channel_importance(class_imp) - Heatmap of class→channel importance
  find_class_specific_channels(class_imp)  - Find channels specific to one class

INTERPRETATION WORKFLOW:
  interpret_single_image(idx)              - Full analysis workflow for one image
  analyze_channel_across_dataset(ch, n)    - How one channel behaves across many images
  analyze_layer1_channel_sources(all_h, ch)- Which L0 channels correlate with L1 channel
  plot_layer1_sources(all_h, ch)           - Visualize L0→L1 connections

BATCH STATISTICS:
  analyze_batch(indices)                   - Analyze multiple samples
  quick_stats(n_samples=50)                - Hoyer sparsity, attribution error across batch
  get_delta_values(all_h)                  - Compute ||D[:,k]|| * |h_k| for each channel

=============================================================================
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from cifar10_patch_rmsnorm import BilinearPatchNetRMSNorm

# %%
# === SETUP ===

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load model
MODEL_PATH = 'cifar10_models/patch_bilinear_rmsnorm_attr10.0_delta1.0.pt'
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
config = checkpoint['config']

model = BilinearPatchNetRMSNorm(
    embed_dim=config['embed_dim'],
    hidden_dim=config['hidden_dim'],
    n_layers=config['n_layers']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(device)

print(f"Loaded model: {MODEL_PATH}")
print(f"Config: {config}")
print(f"Best accuracy: {checkpoint['best_acc']:.2f}%")

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Model dimensions
n_patches = 64  # 8x8 patches
hidden_dim = config['hidden_dim']  # 128
embed_dim = config['embed_dim']  # 48
n_layers = config['n_layers']  # 2

print(f"\nModel: {n_layers} layers, {n_patches} patches, {hidden_dim} hidden dim")

# %%
# === CORE FUNCTIONS ===

def get_attribution(x, y, return_intermediates=False):
    """
    Compute frozen-RMS attribution for a single sample.

    Args:
        x: image tensor [3, 32, 32]
        y: class label (int)
        return_intermediates: if True, return more details

    Returns:
        full_logit: scalar logit for class y
        all_attr: list of [n_patches, hidden_dim] attribution tensors per layer
        all_h: list of [n_patches, hidden_dim] activation tensors per layer
    """
    x_in = x.unsqueeze(0).to(device)

    # First pass: collect RMS values
    with torch.no_grad():
        residual = model.patch_embed(x_in).squeeze(0)
        rms_values = []
        for block in model.blocks:
            mlp_out, h = block.mlp(residual)
            residual = residual + mlp_out
            rms = (residual.pow(2).mean(dim=-1, keepdim=True)).sqrt() + block.norm.eps
            rms_values.append(rms.clone())
            residual = block.norm(residual)

    # Second pass: frozen RMS attribution
    residual = model.patch_embed(x_in).squeeze(0)
    all_h = []

    for li, block in enumerate(model.blocks):
        mlp_out, h = block.mlp(residual)
        h = h.detach().requires_grad_(True)
        all_h.append(h)

        mlp_out = block.mlp.D(h)
        residual = residual + mlp_out

        # Frozen RMS normalization
        frozen_rms = rms_values[li].detach()
        gamma = block.norm.gamma
        residual = residual * (gamma / frozen_rms)

    x_pooled = residual.mean(dim=0)
    logit = model.head.weight[y] @ x_pooled
    if model.head.bias is not None:
        logit = logit + model.head.bias[y]

    # Backward
    logit.backward()

    # Collect attributions
    all_attr = []
    for h in all_h:
        attr = (h.grad * h.detach())  # [n_patches, hidden_dim]
        all_attr.append(attr.detach())

    all_h_detached = [h.detach() for h in all_h]

    if return_intermediates:
        return logit.item(), all_attr, all_h_detached, rms_values, residual.detach()

    return logit.item(), all_attr, all_h_detached


def get_all_logits(x):
    """Get logits for all classes."""
    x_in = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_in)
    return logits.squeeze(0).cpu()


def ablate_channel(x, y, layer_idx, patch_idx, channel_idx):
    """
    Compute logit with ONE channel zeroed out.

    Returns: ablated logit (scalar)
    """
    x_in = x.unsqueeze(0).to(device)

    with torch.no_grad():
        residual = model.patch_embed(x_in).squeeze(0)

        for li, block in enumerate(model.blocks):
            mlp_out, h = block.mlp(residual)

            if li == layer_idx:
                h[patch_idx, channel_idx] = 0.0
                mlp_out = block.mlp.D(h)

            residual = residual + mlp_out
            residual = block.norm(residual)

        x_pooled = residual.mean(dim=0)
        logit = model.head.weight[y] @ x_pooled
        if model.head.bias is not None:
            logit = logit + model.head.bias[y]

    return logit.item()


def ablate_multiple_channels(x, y, channels_to_ablate):
    """
    Ablate multiple channels at once.

    Args:
        channels_to_ablate: list of (layer_idx, patch_idx, channel_idx) tuples

    Returns: ablated logit
    """
    x_in = x.unsqueeze(0).to(device)

    # Group by layer for efficiency
    ablations_by_layer = {0: [], 1: []}
    for (li, pi, ci) in channels_to_ablate:
        ablations_by_layer[li].append((pi, ci))

    with torch.no_grad():
        residual = model.patch_embed(x_in).squeeze(0)

        for li, block in enumerate(model.blocks):
            mlp_out, h = block.mlp(residual)

            for (pi, ci) in ablations_by_layer.get(li, []):
                h[pi, ci] = 0.0

            mlp_out = block.mlp.D(h)
            residual = residual + mlp_out
            residual = block.norm(residual)

        x_pooled = residual.mean(dim=0)
        logit = model.head.weight[y] @ x_pooled
        if model.head.bias is not None:
            logit = logit + model.head.bias[y]

    return logit.item()


def get_top_channels(all_attr, k=20):
    """
    Get top-k channels by attribution magnitude.

    Returns: list of dicts with layer, patch, channel, attr
    """
    # Flatten all attributions
    contribs = torch.cat([a.reshape(-1) for a in all_attr])
    n_per_layer = n_patches * hidden_dim

    sorted_idx = contribs.abs().argsort(descending=True)[:k]

    results = []
    for flat_idx in sorted_idx:
        flat_idx = flat_idx.item()
        layer_idx = flat_idx // n_per_layer
        remainder = flat_idx % n_per_layer
        patch_idx = remainder // hidden_dim
        channel_idx = remainder % hidden_dim

        results.append({
            'layer': layer_idx,
            'patch': patch_idx,
            'channel': channel_idx,
            'attr': contribs[flat_idx].item(),
            'flat_idx': flat_idx,
        })

    return results


def verify_attribution(x, y, top_k=10):
    """
    Verify that attribution matches ablation effect for top-k channels.
    """
    full_logit, all_attr, all_h = get_attribution(x, y)
    top_channels = get_top_channels(all_attr, k=top_k)

    results = []
    for ch in top_channels:
        ablated_logit = ablate_channel(x, y, ch['layer'], ch['patch'], ch['channel'])
        ablation_effect = full_logit - ablated_logit

        results.append({
            **ch,
            'ablation_effect': ablation_effect,
            'error': abs(ch['attr'] - ablation_effect),
        })

    return results, full_logit


# %%
# === VISUALIZATION FUNCTIONS ===

def show_image(x, title=None):
    """Display a CIFAR image (undoing normalization)."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    img = x.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


def plot_attribution_heatmap(all_attr, layer=0, title=None):
    """Plot attribution as 8x8 heatmap (summing over channels)."""
    attr = all_attr[layer]  # [64, 128]

    # Sum over channels to get per-patch attribution
    patch_attr = attr.sum(dim=1).cpu().numpy()  # [64]
    heatmap = patch_attr.reshape(8, 8)

    plt.figure(figsize=(5, 4))
    plt.imshow(heatmap, cmap='RdBu_r', aspect='equal')
    plt.colorbar(label='Attribution')
    plt.title(title or f'Layer {layer} Attribution Heatmap')
    plt.xlabel('Patch X')
    plt.ylabel('Patch Y')
    plt.show()

    return heatmap


def plot_channel_histogram(all_attr, layer=0):
    """Histogram of channel attributions."""
    attr = all_attr[layer].reshape(-1).cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.hist(attr, bins=100, alpha=0.7)
    plt.xlabel('Attribution')
    plt.ylabel('Count')
    plt.title(f'Layer {layer} Attribution Distribution')
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.show()


# %%
# === ANALYSIS: Single Sample ===

# Pick a sample
sample_idx = 0
x, y = test_dataset[sample_idx]

# Get prediction
logits = get_all_logits(x)
pred = logits.argmax().item()
print(f"Sample {sample_idx}: True={CLASSES[y]}, Pred={CLASSES[pred]}")
print(f"Logits: {logits.numpy().round(2)}")

# Show image
show_image(x, f"Sample {sample_idx}: {CLASSES[y]}")

# %%
# Get attribution for correct class
full_logit, all_attr, all_h = get_attribution(x, y)

print(f"\nAttribution for class '{CLASSES[y]}' (logit={full_logit:.3f}):")
print(f"  Layer 0: total={all_attr[0].sum():.3f}, pos={all_attr[0][all_attr[0]>0].sum():.3f}, neg={all_attr[0][all_attr[0]<0].sum():.3f}")
print(f"  Layer 1: total={all_attr[1].sum():.3f}, pos={all_attr[1][all_attr[1]>0].sum():.3f}, neg={all_attr[1][all_attr[1]<0].sum():.3f}")

# %%
# Top channels
top_channels = get_top_channels(all_attr, k=20)

print("\nTop 20 channels by attribution magnitude:")
print(f"{'L':>2} {'P':>3} {'Ch':>4} {'Attr':>10}")
print("-" * 25)
for ch in top_channels:
    print(f"{ch['layer']:>2} {ch['patch']:>3} {ch['channel']:>4} {ch['attr']:>10.4f}")

# %%
# Verify attribution accuracy
results, full_logit = verify_attribution(x, y, top_k=10)

print("\nAttribution vs Ablation (top 10):")
print(f"{'L':>2} {'P':>3} {'Ch':>4} {'Attr':>10} {'Ablation':>10} {'Error':>10}")
print("-" * 50)
for r in results:
    print(f"{r['layer']:>2} {r['patch']:>3} {r['channel']:>4} {r['attr']:>10.4f} {r['ablation_effect']:>10.4f} {r['error']:>10.6f}")

mean_error = np.mean([r['error'] for r in results])
print(f"\nMean attribution error: {mean_error:.6f}")

# %%
# Visualize attribution heatmaps
plot_attribution_heatmap(all_attr, layer=0, title=f"Layer 0 Attribution - {CLASSES[y]}")
plot_attribution_heatmap(all_attr, layer=1, title=f"Layer 1 Attribution - {CLASSES[y]}")

# %%
# === BATCH ANALYSIS ===

def analyze_batch(indices, class_filter=None):
    """Analyze multiple samples."""
    results = []

    for idx in indices:
        x, y = test_dataset[idx]

        if class_filter is not None and y != class_filter:
            continue

        logits = get_all_logits(x)
        pred = logits.argmax().item()
        full_logit, all_attr, all_h = get_attribution(x, y)

        top = get_top_channels(all_attr, k=5)

        results.append({
            'idx': idx,
            'true_class': y,
            'pred_class': pred,
            'correct': y == pred,
            'logit': full_logit,
            'top_channels': top,
            'all_attr': all_attr,
        })

    return results

# Analyze first 50 samples
batch_results = analyze_batch(range(50))

correct = sum(r['correct'] for r in batch_results)
print(f"Accuracy on first 50: {correct}/{len(batch_results)} = {100*correct/len(batch_results):.1f}%")

# %%
# === EXPLORE: Change sample_idx and re-run above cells ===
sample_idx = 5  # Change this!

# %%
# === HELPER: Get D column norms and delta values ===

def get_delta_values(all_h):
    """Compute delta = ||D[:,k]|| * |h_k| for each channel."""
    deltas = []
    for li, h in enumerate(all_h):
        D = model.blocks[li].mlp.D.weight  # [embed_dim, hidden_dim]
        D_norms = D.norm(dim=0)  # [hidden_dim]
        delta = D_norms * h.abs()  # [n_patches, hidden_dim]
        deltas.append(delta)
    return deltas

x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)
deltas = get_delta_values(all_h)

print("Delta values (||D[:,k]|| * |h_k|):")
for li, d in enumerate(deltas):
    print(f"  Layer {li}: max={d.max():.4f}, mean={d.mean():.4f}, >0.1: {(d>0.1).sum().item()}")

# %%
# =============================================================================
# DISTRIBUTION PLOTS
# =============================================================================

def plot_channel_distributions(all_attr, all_h, layer=0, channels=None, n_channels=20, n_per_plot=5):
    """
    Plot distributions of attributions and activations for channels.

    Args:
        all_attr, all_h: from get_attribution()
        layer: which layer
        channels: list of specific channel indices to plot, or None for top channels
        n_channels: how many channels total (if channels=None, uses top by attribution)
        n_per_plot: how many channels per subplot (overlaid with alpha)

    Example:
        # Top 20 channels, 5 per plot (4 plots)
        plot_channel_distributions(all_attr, all_h, layer=0)

        # Specific channels
        plot_channel_distributions(all_attr, all_h, channels=[0, 5, 10, 15, 20])

        # Top 30, 10 per plot
        plot_channel_distributions(all_attr, all_h, n_channels=30, n_per_plot=10)
    """
    attr = all_attr[layer]  # [n_patches, hidden_dim]
    h = all_h[layer]

    if channels is None:
        # Get top channels by attribution magnitude (summed over patches)
        channel_attr_sum = attr.abs().sum(dim=0)  # [hidden_dim]
        channels = channel_attr_sum.argsort(descending=True)[:n_channels].cpu().numpy()
    else:
        channels = np.array(channels)

    n_total = len(channels)
    n_plots = (n_total + n_per_plot - 1) // n_per_plot
    colors = plt.cm.tab10(np.linspace(0, 1, n_per_plot))

    # Attribution distributions
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 4*n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for plot_idx in range(n_plots):
        start = plot_idx * n_per_plot
        end = min(start + n_per_plot, n_total)
        plot_channels = channels[start:end]

        # Attribution histogram
        ax = axes[plot_idx, 0]
        for i, ch in enumerate(plot_channels):
            attr_ch = attr[:, ch].cpu().numpy()
            ax.hist(attr_ch, bins=30, alpha=0.4, color=colors[i], label=f'Ch {ch}')
        ax.set_xlabel('Attribution')
        ax.set_ylabel('Count')
        ax.set_title(f'Layer {layer} Attribution Distribution')
        ax.legend(fontsize=8)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)

        # Activation histogram
        ax = axes[plot_idx, 1]
        for i, ch in enumerate(plot_channels):
            h_ch = h[:, ch].cpu().numpy()
            ax.hist(h_ch, bins=30, alpha=0.4, color=colors[i], label=f'Ch {ch}')
        ax.set_xlabel('Activation (h)')
        ax.set_ylabel('Count')
        ax.set_title(f'Layer {layer} Activation Distribution')
        ax.legend(fontsize=8)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return channels


def plot_attribution_vs_activation_scatter(all_attr, all_h, layer=0, n_channels=5):
    """
    Scatter plot of attribution vs activation for top channels.
    """
    attr = all_attr[layer]
    h = all_h[layer]

    channel_attr_sum = attr.abs().sum(dim=0)
    top_channels = channel_attr_sum.argsort(descending=True)[:n_channels].cpu().numpy()

    fig, axes = plt.subplots(1, n_channels, figsize=(4*n_channels, 4))
    if n_channels == 1:
        axes = [axes]

    for i, ch in enumerate(top_channels):
        ax = axes[i]
        attr_ch = attr[:, ch].cpu().numpy()
        h_ch = h[:, ch].cpu().numpy()

        ax.scatter(h_ch, attr_ch, alpha=0.5, s=20)
        ax.set_xlabel('h (activation)')
        ax.set_ylabel('Attribution')
        ax.set_title(f'Channel {ch}')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


# %%
# =============================================================================
# FIRING MASKS
# =============================================================================

def compute_firing_masks(all_h, method='threshold', threshold=0.1, percentile=90):
    """
    Compute binary firing masks from activations.

    Methods:
    - 'threshold': |h| > threshold
    - 'percentile': |h| > percentile of |h| distribution
    - 'sign': h > 0 (positive firing)
    - 'significant': |h| > mean + std

    Returns: list of [n_patches, hidden_dim] boolean tensors
    """
    masks = []
    for h in all_h:
        h_abs = h.abs()

        if method == 'threshold':
            mask = h_abs > threshold
        elif method == 'percentile':
            thresh = torch.quantile(h_abs.flatten(), percentile / 100.0)
            mask = h_abs > thresh
        elif method == 'sign':
            mask = h > 0
        elif method == 'significant':
            mean = h_abs.mean()
            std = h_abs.std()
            mask = h_abs > (mean + std)
        else:
            raise ValueError(f"Unknown method: {method}")

        masks.append(mask)

    return masks


def compute_firing_rate(masks):
    """Compute firing rate per channel (fraction of patches where it fires)."""
    rates = []
    for mask in masks:
        rate = mask.float().mean(dim=0)  # [hidden_dim]
        rates.append(rate)
    return rates


# %%
# =============================================================================
# CO-FIRING CORRELATION
# =============================================================================

def compute_cofire_correlation(all_h, layer=0, method='continuous'):
    """
    Compute co-firing correlation matrix between channels.

    Methods:
    - 'continuous': Pearson correlation of activations
    - 'binary': Correlation of binary firing masks
    - 'jaccard': Jaccard similarity of firing masks

    Returns: [hidden_dim, hidden_dim] correlation matrix
    """
    h = all_h[layer]  # [n_patches, hidden_dim]

    if method == 'continuous':
        # Pearson correlation
        h_centered = h - h.mean(dim=0, keepdim=True)
        h_norm = h_centered / (h_centered.std(dim=0, keepdim=True) + 1e-8)
        corr = (h_norm.T @ h_norm) / h.shape[0]

    elif method == 'binary':
        # Binary correlation
        masks = compute_firing_masks([h], method='significant')[0].float()
        masks_centered = masks - masks.mean(dim=0, keepdim=True)
        masks_norm = masks_centered / (masks_centered.std(dim=0, keepdim=True) + 1e-8)
        corr = (masks_norm.T @ masks_norm) / h.shape[0]

    elif method == 'jaccard':
        # Jaccard similarity
        masks = compute_firing_masks([h], method='significant')[0].float()
        intersection = masks.T @ masks  # [hidden, hidden]
        union = masks.sum(dim=0, keepdim=True) + masks.sum(dim=0).unsqueeze(1) - intersection
        corr = intersection / (union + 1e-8)

    else:
        raise ValueError(f"Unknown method: {method}")

    return corr.cpu()


def compute_signed_cofire(all_h, layer=0):
    """
    Compute signed co-firing: do channels fire with same sign or opposite?

    Returns:
    - same_sign: [hidden, hidden] - correlation when both positive or both negative
    - opp_sign: [hidden, hidden] - correlation when opposite signs
    """
    h = all_h[layer]  # [n_patches, hidden_dim]

    # Get signs
    signs = (h > 0).float() * 2 - 1  # +1 or -1

    # Same sign: both positive or both negative
    same_sign_mask = (signs.unsqueeze(2) == signs.unsqueeze(1)).float()  # [patches, h1, h2]

    # Correlation weighted by same-sign co-occurrence
    h_abs = h.abs()
    h_abs_outer = h_abs.unsqueeze(2) * h_abs.unsqueeze(1)  # [patches, h1, h2]

    same_sign_corr = (same_sign_mask * h_abs_outer).sum(dim=0) / (same_sign_mask.sum(dim=0) + 1e-8)
    opp_sign_corr = ((1 - same_sign_mask) * h_abs_outer).sum(dim=0) / ((1 - same_sign_mask).sum(dim=0) + 1e-8)

    return same_sign_corr.cpu(), opp_sign_corr.cpu()


def plot_cofire_matrix(corr, title='Co-firing Correlation', top_k=50):
    """Plot co-firing correlation matrix for top-k channels."""
    # Find top-k channels by mean absolute correlation
    mean_corr = corr.abs().mean(dim=1)
    top_idx = mean_corr.argsort(descending=True)[:top_k]

    corr_subset = corr[top_idx][:, top_idx]

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_subset.numpy(), cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title(f'{title} (top {top_k} channels)')
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    plt.show()

    return top_idx


# %%
# =============================================================================
# CAUSAL CONNECTIVITY (from weights, no data needed)
# =============================================================================

def compute_causal_connectivity():
    """
    Compute causal connectivity from model weights.

    For bilinear layer: out = D @ (L(x) * R(x))
    Channel k influences output via D[:, k]

    Returns dict with:
    - D_col_norms: [n_layers, hidden_dim] - importance of each channel
    - channel_to_class: [n_layers, hidden_dim, n_classes] - direct effect on each class
    """
    results = {
        'D_col_norms': [],
        'channel_to_class': [],
        'L_row_norms': [],
        'R_row_norms': [],
    }

    # Unembed weight
    W_u = model.head.weight  # [n_classes, embed_dim]

    for li, block in enumerate(model.blocks):
        D = block.mlp.D.weight  # [embed_dim, hidden_dim]
        L = block.mlp.L.weight  # [hidden_dim, embed_dim]
        R = block.mlp.R.weight  # [hidden_dim, embed_dim]

        # Column norms of D (how much each channel can affect residual)
        D_col_norms = D.norm(dim=0)  # [hidden_dim]
        results['D_col_norms'].append(D_col_norms.detach().cpu())

        # Direct logit effect: W_u @ D[:, k] for each channel k
        # This is how much channel k directly contributes to each class logit
        # (ignoring RMS norm scaling)
        channel_to_class = W_u @ D  # [n_classes, hidden_dim]
        results['channel_to_class'].append(channel_to_class.detach().cpu())

        # Row norms of L and R (how sensitive each channel is to input)
        L_row_norms = L.norm(dim=1)  # [hidden_dim]
        R_row_norms = R.norm(dim=1)  # [hidden_dim]
        results['L_row_norms'].append(L_row_norms.detach().cpu())
        results['R_row_norms'].append(R_row_norms.detach().cpu())

    return results


def plot_channel_class_effects(connectivity, layer=0, top_k=20):
    """
    Plot which classes each channel promotes/suppresses.
    (Original heatmap version)
    """
    channel_to_class = connectivity['channel_to_class'][layer]  # [n_classes, hidden_dim]

    # Find top channels by max effect on any class
    max_effect = channel_to_class.abs().max(dim=0).values
    top_channels = max_effect.argsort(descending=True)[:top_k]

    subset = channel_to_class[:, top_channels].numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(subset, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Logit contribution')
    plt.xlabel('Channel (sorted by max effect)')
    plt.ylabel('Class')
    plt.yticks(range(10), CLASSES)
    plt.xticks(range(top_k), top_channels.numpy(), rotation=45)
    plt.title(f'Layer {layer}: Channel → Class Effects (top {top_k})')
    plt.tight_layout()
    plt.show()

    return top_channels


def plot_top_channels_per_class(connectivity, layer=0, top_k=10):
    """
    For each class, show the top-k most positive and top-k most negative channels.

    Creates a grid where each row is a class, showing:
    - Left side: top-k channels that PROMOTE this class (positive effect)
    - Right side: top-k channels that SUPPRESS this class (negative effect)

    Each cell shows channel ID and effect value.
    """
    channel_to_class = connectivity['channel_to_class'][layer]  # [n_classes, hidden_dim]

    fig, axes = plt.subplots(10, 2, figsize=(16, 20))

    results = {}

    for class_idx in range(10):
        class_name = CLASSES[class_idx]
        effects = channel_to_class[class_idx]  # [hidden_dim]

        # Top positive (promote this class)
        top_pos_idx = effects.argsort(descending=True)[:top_k]
        top_pos_vals = effects[top_pos_idx]

        # Top negative (suppress this class)
        top_neg_idx = effects.argsort()[:top_k]
        top_neg_vals = effects[top_neg_idx]

        results[class_name] = {
            'promote': list(zip(top_pos_idx.tolist(), top_pos_vals.tolist())),
            'suppress': list(zip(top_neg_idx.tolist(), top_neg_vals.tolist())),
        }

        # Plot positive channels
        ax = axes[class_idx, 0]
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, top_k))
        bars = ax.barh(range(top_k), top_pos_vals.numpy(), color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f'Ch {ch}' for ch in top_pos_idx.numpy()])
        ax.set_xlabel('Effect')
        ax.set_title(f'{class_name}: PROMOTE', fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, (ch, val) in enumerate(zip(top_pos_idx, top_pos_vals)):
            ax.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=8)

        # Plot negative channels
        ax = axes[class_idx, 1]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, top_k))
        bars = ax.barh(range(top_k), top_neg_vals.numpy(), color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f'Ch {ch}' for ch in top_neg_idx.numpy()])
        ax.set_xlabel('Effect')
        ax.set_title(f'{class_name}: SUPPRESS', fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, (ch, val) in enumerate(zip(top_neg_idx, top_neg_vals)):
            ax.text(val - 0.01, i, f'{val:.2f}', va='center', ha='right', fontsize=8)

    plt.suptitle(f'Layer {layer}: Top Channels per Class (from weights)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    return results


def print_top_channels_per_class(connectivity, layer=0, top_k=5):
    """
    Print top channels per class in a readable format.
    """
    channel_to_class = connectivity['channel_to_class'][layer]

    print(f"\nLayer {layer}: Top {top_k} channels per class (from W_u @ D)")
    print("=" * 70)

    for class_idx in range(10):
        class_name = CLASSES[class_idx]
        effects = channel_to_class[class_idx]

        top_pos_idx = effects.argsort(descending=True)[:top_k]
        top_neg_idx = effects.argsort()[:top_k]

        pos_str = ', '.join([f'{ch}({effects[ch]:.2f})' for ch in top_pos_idx])
        neg_str = ', '.join([f'{ch}({effects[ch]:.2f})' for ch in top_neg_idx])

        print(f"\n{class_name:12s}")
        print(f"  Promote:  {pos_str}")
        print(f"  Suppress: {neg_str}")


# %%
# =============================================================================
# SPATIAL PATTERNS
# =============================================================================
#
# EXPLANATION: The image is divided into 8x8 = 64 patches. Each channel has an
# activation value at each patch. The "spatial pattern" shows WHERE in the image
# a channel fires.
#
# Example interpretations:
# - A channel that fires only in top-left patches might detect "top-left corner features"
# - A channel that fires uniformly everywhere detects "global features"
# - A channel with high activation in center might detect "center of object"
#
# This helps understand if channels are spatially localized or global.

def analyze_spatial_patterns(all_h, layer=0):
    """
    Analyze spatial activation patterns across the 8x8 patch grid.

    The image is 32x32 pixels divided into 8x8 = 64 patches (4x4 pixels each).
    Each channel has an activation at each patch. This function analyzes
    WHERE in the image each channel fires.

    Returns dict with:
    - patch_variance: [hidden_dim] - how much activation varies across patches
      (high = spatially selective, low = fires everywhere equally)
    - y_corr: [hidden_dim] - correlation with vertical position
      (positive = fires more at bottom, negative = fires more at top)
    - x_corr: [hidden_dim] - correlation with horizontal position
      (positive = fires more at right, negative = fires more at left)
    - locality: [hidden_dim] - how concentrated the activation is
      (high = fires in small region, low = fires everywhere)
    - y_cm, x_cm: [hidden_dim] - center of mass of activation (0-7 range)
    """
    h = all_h[layer]  # [n_patches=64, hidden_dim=128]

    # Reshape to spatial grid
    h_spatial = h.reshape(8, 8, -1)  # [8, 8, hidden_dim]

    # Variance across patches (high = spatially varying)
    patch_variance = h.var(dim=0)  # [hidden_dim]

    # Correlation with position
    y_pos = torch.arange(8).float().to(device).view(8, 1, 1).expand(8, 8, 1)
    x_pos = torch.arange(8).float().to(device).view(1, 8, 1).expand(8, 8, 1)

    h_centered = h_spatial - h_spatial.mean(dim=(0, 1), keepdim=True)
    y_centered = y_pos - y_pos.mean()
    x_centered = x_pos - x_pos.mean()

    # Correlation with y position
    y_corr = (h_centered * y_centered).mean(dim=(0, 1)) / (h_centered.std(dim=(0, 1)) * y_centered.std() + 1e-8)
    x_corr = (h_centered * x_centered).mean(dim=(0, 1)) / (h_centered.std(dim=(0, 1)) * x_centered.std() + 1e-8)

    # Locality: inverse of spatial spread (high = fires in small region)
    h_abs = h_spatial.abs()
    total_mass = h_abs.sum(dim=(0, 1))  # [hidden_dim]

    # Center of mass
    y_cm = (h_abs * y_pos).sum(dim=(0, 1)) / (total_mass + 1e-8)
    x_cm = (h_abs * x_pos).sum(dim=(0, 1)) / (total_mass + 1e-8)

    # Spread (variance around center of mass)
    y_spread = (h_abs * (y_pos.squeeze(-1) - y_cm) ** 2).sum(dim=(0, 1)) / (total_mass + 1e-8)
    x_spread = (h_abs * (x_pos.squeeze(-1) - x_cm) ** 2).sum(dim=(0, 1)) / (total_mass + 1e-8)

    locality = 1.0 / (y_spread + x_spread + 1e-8)

    return {
        'patch_variance': patch_variance.cpu(),
        'y_corr': y_corr.cpu(),
        'x_corr': x_corr.cpu(),
        'locality': locality.cpu(),
        'y_cm': y_cm.cpu(),
        'x_cm': x_cm.cpu(),
    }


def plot_channel_spatial_pattern(all_h, layer=0, channel=0):
    """
    Plot WHERE in the image a channel fires (8x8 heatmap).

    The 32x32 image is divided into 8x8 patches. This shows the channel's
    activation at each patch location.

    Red = positive activation (channel fires)
    Blue = negative activation
    White = near zero

    Example: If a channel detects "wheels", you'd see high activation
    at the bottom of car images where wheels typically appear.
    """
    h = all_h[layer]  # [64, 128]
    h_ch = h[:, channel].reshape(8, 8).cpu().numpy()

    plt.figure(figsize=(5, 4))
    vmax = max(abs(h_ch.min()), abs(h_ch.max()))
    plt.imshow(h_ch, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
    plt.colorbar(label='Activation')
    plt.title(f'Layer {layer}, Channel {channel}\n(where in image it fires)')
    plt.xlabel('X (left→right)')
    plt.ylabel('Y (top→bottom)')
    plt.show()

    return h_ch


def plot_multiple_spatial_patterns(all_h, layer=0, channels=None, n_channels=16):
    """
    Plot spatial patterns for multiple channels in a grid.
    """
    h = all_h[layer]

    if channels is None:
        # Use channels with highest variance (most spatially interesting)
        variance = h.var(dim=0)
        channels = variance.argsort(descending=True)[:n_channels].cpu().numpy()
    else:
        channels = np.array(channels)

    n = len(channels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).flatten()

    for i, ch in enumerate(channels):
        ax = axes[i]
        h_ch = h[:, ch].reshape(8, 8).cpu().numpy()
        vmax = max(abs(h_ch.min()), abs(h_ch.max()), 0.01)
        im = ax.imshow(h_ch, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Ch {ch}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for i in range(len(channels), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Layer {layer}: Spatial Activation Patterns', fontsize=12)
    plt.tight_layout()
    plt.show()

    return channels


# %%
# =============================================================================
# CHANNEL CLUSTERING
# =============================================================================

def cluster_channels(all_attr, all_h, layer=0, n_clusters=10, feature='activation'):
    """
    Cluster channels by their behavior.

    Features:
    - 'activation': Cluster by activation patterns
    - 'attribution': Cluster by attribution patterns
    - 'combined': Use both

    Returns: cluster assignments and cluster centers
    """
    from sklearn.cluster import KMeans

    if feature == 'activation':
        data = all_h[layer].T.cpu().numpy()  # [hidden_dim, n_patches]
    elif feature == 'attribution':
        data = all_attr[layer].T.cpu().numpy()
    elif feature == 'combined':
        h = all_h[layer].T.cpu().numpy()
        a = all_attr[layer].T.cpu().numpy()
        data = np.concatenate([h, a], axis=1)
    else:
        raise ValueError(f"Unknown feature: {feature}")

    # Normalize
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)

    return labels, kmeans.cluster_centers_


def analyze_clusters(labels, connectivity, layer=0):
    """
    Analyze what each cluster represents.
    """
    channel_to_class = connectivity['channel_to_class'][layer]  # [n_classes, hidden_dim]
    D_norms = connectivity['D_col_norms'][layer]

    n_clusters = labels.max() + 1

    print(f"\nCluster Analysis (Layer {layer}):")
    print("-" * 60)

    for c in range(n_clusters):
        mask = labels == c
        n_channels = mask.sum()

        # Average class effects for this cluster
        cluster_class_effects = channel_to_class[:, mask].mean(dim=1)
        top_class = cluster_class_effects.argmax().item()
        bot_class = cluster_class_effects.argmin().item()

        # Average D norm
        avg_d_norm = D_norms[mask].mean().item()

        print(f"\nCluster {c}: {n_channels} channels")
        print(f"  Promotes: {CLASSES[top_class]} ({cluster_class_effects[top_class]:.3f})")
        print(f"  Suppresses: {CLASSES[bot_class]} ({cluster_class_effects[bot_class]:.3f})")
        print(f"  Avg D norm: {avg_d_norm:.4f}")


# %%
# =============================================================================
# FEATURE GROUPS BY CLASS
# =============================================================================

def find_feature_groups_by_class(n_samples=100):
    """
    Find which channels are most important for each class.

    Returns: [n_classes, hidden_dim, n_layers] importance matrix
    """
    class_importance = torch.zeros(10, hidden_dim, n_layers)
    class_counts = torch.zeros(10)

    # Sample from each class
    indices_by_class = {c: [] for c in range(10)}
    for idx in range(len(test_dataset)):
        _, y = test_dataset[idx]
        if len(indices_by_class[y]) < n_samples // 10:
            indices_by_class[y].append(idx)
        if all(len(v) >= n_samples // 10 for v in indices_by_class.values()):
            break

    # Collect attributions
    for c in range(10):
        for idx in indices_by_class[c]:
            x, y = test_dataset[idx]
            _, all_attr, _ = get_attribution(x, y)

            for li, attr in enumerate(all_attr):
                # Sum over patches
                channel_importance = attr.abs().sum(dim=0).cpu()  # [hidden_dim]
                class_importance[c, :, li] += channel_importance

            class_counts[c] += 1

    # Normalize
    class_importance = class_importance / class_counts.view(-1, 1, 1)

    return class_importance


def plot_class_channel_importance(class_importance, layer=0, top_k=20):
    """
    Plot which channels are most important for each class.
    """
    importance = class_importance[:, :, layer]  # [n_classes, hidden_dim]

    # Find top channels overall
    total_importance = importance.sum(dim=0)
    top_channels = total_importance.argsort(descending=True)[:top_k]

    subset = importance[:, top_channels].numpy()

    plt.figure(figsize=(14, 6))
    plt.imshow(subset, cmap='viridis', aspect='auto')
    plt.colorbar(label='Importance')
    plt.xlabel('Channel (sorted by total importance)')
    plt.ylabel('Class')
    plt.yticks(range(10), CLASSES)
    plt.xticks(range(top_k), top_channels.numpy(), rotation=45)
    plt.title(f'Layer {layer}: Channel Importance by Class (top {top_k})')
    plt.tight_layout()
    plt.show()

    return top_channels


def find_class_specific_channels(class_importance, layer=0, threshold=2.0):
    """
    Find channels that are specific to certain classes.

    Returns channels where importance for one class is > threshold * average.
    """
    importance = class_importance[:, :, layer]  # [n_classes, hidden_dim]

    mean_importance = importance.mean(dim=0, keepdim=True)
    specificity = importance / (mean_importance + 1e-8)

    specific_channels = {}
    for c in range(10):
        channels = (specificity[c] > threshold).nonzero().squeeze(-1).tolist()
        if isinstance(channels, int):
            channels = [channels]
        specific_channels[CLASSES[c]] = channels

    return specific_channels


# %%
# =============================================================================
# INTERPRETATION WORKFLOW
# =============================================================================

def interpret_single_image(idx):
    """
    Full interpretation workflow for a single image.
    """
    x, y = test_dataset[idx]

    # Get predictions
    logits = get_all_logits(x)
    pred = logits.argmax().item()

    print(f"=" * 60)
    print(f"Sample {idx}: True={CLASSES[y]}, Pred={CLASSES[pred]}")
    print(f"Logits: {logits.numpy().round(2)}")
    print(f"=" * 60)

    # Show image
    show_image(x, f"Sample {idx}: {CLASSES[y]}")

    # Get attribution
    full_logit, all_attr, all_h = get_attribution(x, y)

    print(f"\nAttribution Summary (for true class '{CLASSES[y]}'):")
    for li in range(n_layers):
        attr = all_attr[li]
        print(f"  Layer {li}: total={attr.sum():.3f}, "
              f"pos={attr[attr>0].sum():.3f}, neg={attr[attr<0].sum():.3f}")

    # Top channels
    top_channels = get_top_channels(all_attr, k=10)

    print(f"\nTop 10 channels by |attribution|:")
    print(f"{'L':>2} {'P':>3} {'Ch':>4} {'Attr':>10} {'h':>10}")
    print("-" * 35)
    for ch in top_channels:
        h_val = all_h[ch['layer']][ch['patch'], ch['channel']].item()
        print(f"{ch['layer']:>2} {ch['patch']:>3} {ch['channel']:>4} "
              f"{ch['attr']:>10.4f} {h_val:>10.4f}")

    # Attribution heatmaps
    for li in range(n_layers):
        plot_attribution_heatmap(all_attr, layer=li, title=f"Layer {li} Attribution - {CLASSES[y]}")

    # Spatial patterns for top channels
    print(f"\nSpatial patterns for top 3 channels:")
    for ch in top_channels[:3]:
        plot_channel_spatial_pattern(all_h, layer=ch['layer'], channel=ch['channel'])

    return full_logit, all_attr, all_h, top_channels


def analyze_channel_across_dataset(channel, layer=0, n_samples=100):
    """
    Analyze how a specific channel behaves across many samples.
    """
    activations = []
    attributions = []
    classes = []

    for idx in range(min(n_samples, len(test_dataset))):
        x, y = test_dataset[idx]
        _, all_attr, all_h = get_attribution(x, y)

        # Get channel values
        h_ch = all_h[layer][:, channel].cpu()  # [n_patches]
        attr_ch = all_attr[layer][:, channel].cpu()

        activations.append(h_ch.mean().item())
        attributions.append(attr_ch.sum().item())
        classes.append(y)

    # Plot by class
    activations = np.array(activations)
    attributions = np.array(attributions)
    classes = np.array(classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for c in range(10):
        mask = classes == c
        ax1.scatter(np.ones(mask.sum()) * c + np.random.randn(mask.sum()) * 0.1,
                   activations[mask], alpha=0.5, s=20, label=CLASSES[c])
        ax2.scatter(np.ones(mask.sum()) * c + np.random.randn(mask.sum()) * 0.1,
                   attributions[mask], alpha=0.5, s=20)

    ax1.set_xticks(range(10))
    ax1.set_xticklabels(CLASSES, rotation=45)
    ax1.set_ylabel('Mean Activation')
    ax1.set_title(f'Layer {layer}, Channel {channel}: Activation by Class')

    ax2.set_xticks(range(10))
    ax2.set_xticklabels(CLASSES, rotation=45)
    ax2.set_ylabel('Total Attribution')
    ax2.set_title(f'Layer {layer}, Channel {channel}: Attribution by Class')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return activations, attributions, classes


# %%
# =============================================================================
# LAYER 1 CHANNEL SOURCES (which Layer 0 channels feed into it)
# =============================================================================

def analyze_layer1_channel_sources(all_h, channel_l1):
    """
    Analyze which Layer 0 channels contribute to a Layer 1 channel.

    For bilinear: h1 = L1(norm(x0 + D0@h0)) * R1(norm(x0 + D0@h0))

    This is complex due to the bilinear interaction. We approximate by looking
    at correlation between L0 channel activations and L1 channel activation.
    """
    h0 = all_h[0]  # [n_patches, hidden_dim]
    h1 = all_h[1]

    h1_ch = h1[:, channel_l1]  # [n_patches]

    # Correlation between each L0 channel and this L1 channel
    h0_centered = h0 - h0.mean(dim=0, keepdim=True)
    h1_centered = h1_ch - h1_ch.mean()

    corr = (h0_centered * h1_centered.unsqueeze(1)).mean(dim=0)
    corr = corr / (h0.std(dim=0) * h1_ch.std() + 1e-8)

    return corr.cpu()


def plot_layer1_sources(all_h, channel_l1, top_k=10):
    """
    Plot which Layer 0 channels correlate with a Layer 1 channel.
    """
    corr = analyze_layer1_channel_sources(all_h, channel_l1)

    # Get top correlated channels
    top_pos = corr.argsort(descending=True)[:top_k]
    top_neg = corr.argsort()[:top_k]

    print(f"\nLayer 1 Channel {channel_l1} sources:")
    print(f"\nTop positive correlations with Layer 0:")
    for ch in top_pos:
        print(f"  Ch {ch.item()}: {corr[ch]:.3f}")

    print(f"\nTop negative correlations with Layer 0:")
    for ch in top_neg:
        print(f"  Ch {ch.item()}: {corr[ch]:.3f}")

    # Bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(top_k), corr[top_pos].numpy())
    ax1.set_xticks(range(top_k))
    ax1.set_xticklabels(top_pos.numpy())
    ax1.set_xlabel('Layer 0 Channel')
    ax1.set_ylabel('Correlation')
    ax1.set_title(f'Top Positive Sources for L1 Ch {channel_l1}')

    ax2.bar(range(top_k), corr[top_neg].numpy())
    ax2.set_xticks(range(top_k))
    ax2.set_xticklabels(top_neg.numpy())
    ax2.set_xlabel('Layer 0 Channel')
    ax2.set_ylabel('Correlation')
    ax2.set_title(f'Top Negative Sources for L1 Ch {channel_l1}')

    plt.tight_layout()
    plt.show()

    return corr


# %%
# =============================================================================
# QUICK STATS
# =============================================================================

def quick_stats(n_samples=50):
    """
    Quick statistics across multiple samples.
    """
    print("Computing statistics across samples...")

    all_hoyers = {0: [], 1: []}
    all_sparsity = {0: [], 1: []}
    attribution_errors = []

    for idx in range(n_samples):
        x, y = test_dataset[idx]
        full_logit, all_attr, all_h = get_attribution(x, y)

        # Hoyer sparsity
        for li in range(n_layers):
            attr_flat = all_attr[li].flatten()
            n = attr_flat.shape[0]
            l1 = attr_flat.abs().sum()
            l2 = (attr_flat ** 2).sum().sqrt()
            hoyer = (n ** 0.5 - l1 / (l2 + 1e-10)) / (n ** 0.5 - 1)
            all_hoyers[li].append(hoyer.item())

            # Sparsity (fraction near zero)
            sparsity = (attr_flat.abs() < 0.01).float().mean().item()
            all_sparsity[li].append(sparsity)

        # Attribution error (verify top-1)
        top = get_top_channels(all_attr, k=1)[0]
        ablated = ablate_channel(x, y, top['layer'], top['patch'], top['channel'])
        error = abs(top['attr'] - (full_logit - ablated))
        attribution_errors.append(error)

    print(f"\nStatistics over {n_samples} samples:")
    for li in range(n_layers):
        print(f"\nLayer {li}:")
        print(f"  Hoyer: mean={np.mean(all_hoyers[li]):.3f}, std={np.std(all_hoyers[li]):.3f}")
        print(f"  Sparsity: mean={np.mean(all_sparsity[li]):.3f}")

    print(f"\nAttribution error (top-1): mean={np.mean(attribution_errors):.6f}, "
          f"max={np.max(attribution_errors):.6f}")

    return {
        'hoyers': all_hoyers,
        'sparsity': all_sparsity,
        'errors': attribution_errors,
    }


# %%
# =============================================================================
# EXAMPLES - Each section is a separate cell, run any one independently
# =============================================================================

# %%
# EXAMPLE 1: Basic attribution for one image
x, y = test_dataset[0]  # Get first test image
show_image(x, f"Class: {CLASSES[y]}")

full_logit, all_attr, all_h = get_attribution(x, y)
print(f"Logit for {CLASSES[y]}: {full_logit:.3f}")
print(f"Layer 0 attribution sum: {all_attr[0].sum():.3f}")
print(f"Layer 1 attribution sum: {all_attr[1].sum():.3f}")

top = get_top_channels(all_attr, k=10)
for ch in top:
    print(f"  L{ch['layer']} P{ch['patch']} Ch{ch['channel']}: {ch['attr']:.4f}")

# %%
# EXAMPLE 2: Verify attribution accuracy (should be <0.1 error)
x, y = test_dataset[0]
results, full_logit = verify_attribution(x, y, top_k=5)
print(f"\nAttribution vs Ablation for top 5 channels:")
for r in results:
    print(f"  Ch {r['channel']}: attr={r['attr']:.4f}, ablation={r['ablation_effect']:.4f}, error={r['error']:.6f}")

# %%
# EXAMPLE 3: Distribution plots (histograms of attr & h)
x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)

# Top 20 channels, 5 per plot (4 subplots)
plot_channel_distributions(all_attr, all_h, layer=0)

# %%
# EXAMPLE 3b: Distribution plots - specific channels
x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)

plot_channel_distributions(all_attr, all_h, channels=[0, 10, 20, 30, 40])

# %%
# EXAMPLE 4: Causal connectivity from weights (no data needed)
connectivity = compute_causal_connectivity()

# Print top channels per class (text)
print_top_channels_per_class(connectivity, layer=0, top_k=5)

# %%
# EXAMPLE 4b: Causal connectivity - bar charts
connectivity = compute_causal_connectivity()

# Bar charts showing promote/suppress channels per class
plot_top_channels_per_class(connectivity, layer=0, top_k=10)

# %%
# EXAMPLE 4c: Causal connectivity - heatmap version
connectivity = compute_causal_connectivity()

plot_channel_class_effects(connectivity, layer=0, top_k=30)

# %%
# EXAMPLE 5: Spatial patterns (where in image does channel fire?)
x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)

# Single channel spatial pattern
plot_channel_spatial_pattern(all_h, layer=0, channel=0)

# %%
# EXAMPLE 5b: Spatial patterns - grid of 16 channels
x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)

plot_multiple_spatial_patterns(all_h, layer=0, n_channels=16)

# %%
# EXAMPLE 5c: Spatial patterns - analyze statistics
x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)

spatial = analyze_spatial_patterns(all_h, layer=0)
print(f"Channel 0: y_corr={spatial['y_corr'][0]:.3f}, x_corr={spatial['x_corr'][0]:.3f}")
print(f"Channel 0: locality={spatial['locality'][0]:.3f}")

# %%
# EXAMPLE 6: Co-firing correlation
x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)

# Continuous correlation (Pearson)
corr = compute_cofire_correlation(all_h, layer=0, method='continuous')
plot_cofire_matrix(corr, title='Continuous Co-firing', top_k=30)

# %%
# EXAMPLE 6b: Co-firing - Jaccard similarity
x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)

corr_jaccard = compute_cofire_correlation(all_h, layer=0, method='jaccard')
plot_cofire_matrix(corr_jaccard, title='Jaccard Co-firing', top_k=30)

# %%
# EXAMPLE 7: Firing masks
x, y = test_dataset[0]
_, _, all_h = get_attribution(x, y)

masks_thresh = compute_firing_masks(all_h, method='threshold', threshold=0.1)
masks_pct = compute_firing_masks(all_h, method='percentile', percentile=90)
masks_sign = compute_firing_masks(all_h, method='sign')  # h > 0

rates = compute_firing_rate(masks_thresh)
print(f"Layer 0 firing rates: min={rates[0].min():.3f}, max={rates[0].max():.3f}")

# %%
# EXAMPLE 8: Clustering channels
x, y = test_dataset[0]
_, all_attr, all_h = get_attribution(x, y)
connectivity = compute_causal_connectivity()

labels, centers = cluster_channels(all_attr, all_h, layer=0, n_clusters=10, feature='activation')
analyze_clusters(labels, connectivity, layer=0)

# %%
# EXAMPLE 9: Feature groups by class (takes ~1 min, analyzes 100 samples)
class_importance = find_feature_groups_by_class(n_samples=100)

plot_class_channel_importance(class_importance, layer=0, top_k=30)

# %%
# EXAMPLE 9b: Find class-specific channels
class_importance = find_feature_groups_by_class(n_samples=100)

specific = find_class_specific_channels(class_importance, layer=0, threshold=2.0)
for cls, channels in specific.items():
    if channels:
        print(f"{cls}: channels {channels}")

# %%
# EXAMPLE 10: Full interpretation workflow for one image
interpret_single_image(0)  # Sample index 0

# %%
# EXAMPLE 11: Analyze one channel across the dataset
activations, attributions, classes = analyze_channel_across_dataset(
    channel=50, layer=0, n_samples=100
)

# %%
# EXAMPLE 12: Layer 0 → Layer 1 connections
x, y = test_dataset[0]
_, _, all_h = get_attribution(x, y)

plot_layer1_sources(all_h, channel_l1=50, top_k=10)

# %%
# EXAMPLE 13: Quick batch statistics
stats = quick_stats(n_samples=50)
print(f"Mean attribution error: {np.mean(stats['errors']):.6f}")

# %%
# EXAMPLE 14: Attribution heatmaps (per-patch attribution)
x, y = test_dataset[0]
_, all_attr, _ = get_attribution(x, y)

plot_attribution_heatmap(all_attr, layer=0, title=f"Layer 0 - {CLASSES[y]}")
plot_attribution_heatmap(all_attr, layer=1, title=f"Layer 1 - {CLASSES[y]}")

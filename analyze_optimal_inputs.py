# %%
"""
Optimal Input Analysis for Bilinear Channels

For each channel, find:
1. Optimal input (what input maximizes this channel's activation)
2. Top activating examples from the dataset

Layer 0: Analytical solution
  h_k = (L_k · x) * (R_k · x) is a quadratic form x^T M x
  where M = (L_k ⊗ R_k^T + R_k ⊗ L_k^T) / 2
  Optimal x is the principal eigenvector of M

Layer 1: Gradient-based optimization
  Must optimize through Layer 0 + RMSNorm, so use gradient descent

=============================================================================
FUNCTION REFERENCE
=============================================================================

LAYER 0 ANALYTICAL:
  compute_layer0_optimal_inputs()      - Compute optimal input for all L0 channels
  get_layer0_optimal_input(channel)    - Get optimal input for one channel
  embed_to_patch(embed_vector)         - Convert embedding to 4x4 RGB patch
  visualize_optimal_as_patch(channel)  - Show optimal inputs as 4x4 RGB patches
  visualize_optimal_input_layer0(ch)   - Show optimal input as embedding bar chart

LAYER 1 OPTIMIZATION:
  optimize_layer1_input(channel)       - Gradient descent to find optimal input
  optimize_layer1_input_batch(channels)- Optimize multiple channels

TOP ACTIVATING EXAMPLES:
  find_top_activating(channel, layer)  - Find test images that most activate channel
  show_top_activating(channel, layer)  - Display top activating images

VISUALIZATION:
  compare_optimal_vs_real(channel, layer) - Compare optimal input to real examples
  analyze_channel_full(channel, layer)    - Full analysis workflow

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
print(f"Config: embed_dim={config['embed_dim']}, hidden_dim={config['hidden_dim']}, n_layers={config['n_layers']}")

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

print(f"Model: {n_layers} layers, {n_patches} patches, {hidden_dim} hidden dim, {embed_dim} embed dim")


# %%
# =============================================================================
# LAYER 0: ANALYTICAL OPTIMAL INPUTS
# =============================================================================
#
# For bilinear layer: h_k = (L_k · x) * (R_k · x)
# This is a quadratic form: h_k = x^T M x
# where M = (L_k ⊗ R_k^T + R_k ⊗ L_k^T) / 2  (symmetrized outer product)
#
# The optimal x (unit norm) is the eigenvector of M with largest eigenvalue.
# The eigenvalue gives the maximum activation value.

def compute_layer0_optimal_inputs():
    """
    Compute optimal input directions for each Layer 0 channel.

    Returns:
        pos_optimal: [hidden_dim, embed_dim] - optimal input for positive activation
        neg_optimal: [hidden_dim, embed_dim] - optimal input for negative activation
        pos_eigenvalues: [hidden_dim] - largest eigenvalue (max positive activation)
        neg_eigenvalues: [hidden_dim] - smallest eigenvalue (max negative activation)
        alignments: [hidden_dim] - cosine similarity between L_k and R_k
    """
    L = model.blocks[0].mlp.L.weight.detach()  # [hidden_dim, embed_dim]
    R = model.blocks[0].mlp.R.weight.detach()  # [hidden_dim, embed_dim]

    pos_optimal = []
    neg_optimal = []
    pos_eigenvalues = []
    neg_eigenvalues = []
    alignments = []

    for k in range(hidden_dim):
        L_k = L[k]
        R_k = R[k]

        # Symmetrized outer product
        M = (torch.outer(L_k, R_k) + torch.outer(R_k, L_k)) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        # eigh returns in ascending order

        # Positive: largest eigenvalue (last)
        pos_ev = eigenvectors[:, -1]
        pos_ev = pos_ev / (pos_ev.norm() + 1e-8)
        pos_optimal.append(pos_ev)
        pos_eigenvalues.append(eigenvalues[-1].item())

        # Negative: smallest eigenvalue (first)
        neg_ev = eigenvectors[:, 0]
        neg_ev = neg_ev / (neg_ev.norm() + 1e-8)
        neg_optimal.append(neg_ev)
        neg_eigenvalues.append(eigenvalues[0].item())

        # L-R alignment
        alignment = F.cosine_similarity(L_k.unsqueeze(0), R_k.unsqueeze(0)).item()
        alignments.append(alignment)

    pos_optimal = torch.stack(pos_optimal).cpu()
    neg_optimal = torch.stack(neg_optimal).cpu()
    pos_eigenvalues = torch.tensor(pos_eigenvalues)
    neg_eigenvalues = torch.tensor(neg_eigenvalues)
    alignments = torch.tensor(alignments)

    return pos_optimal, neg_optimal, pos_eigenvalues, neg_eigenvalues, alignments


def get_layer0_optimal_input(channel):
    """
    Get optimal input for a single Layer 0 channel.

    For h_k = (L_k · x)(R_k · x), this is a quadratic form x^T M x where
    M = (L_k ⊗ R_k^T + R_k ⊗ L_k^T) / 2 (symmetrized outer product).

    Key insight: h(-x) = h(x) since both dot products flip sign.

    Returns two optimal directions:
    - positive_optimal: eigenvector with largest positive eigenvalue (max h_k > 0)
    - negative_optimal: eigenvector with most negative eigenvalue (min h_k < 0)

    These are NOT inverses of each other - they are different eigenvectors.
    """
    L = model.blocks[0].mlp.L.weight.detach()
    R = model.blocks[0].mlp.R.weight.detach()

    L_k = L[channel]
    R_k = R[channel]

    # Symmetrized outer product
    M = (torch.outer(L_k, R_k) + torch.outer(R_k, L_k)) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    # eigh returns eigenvalues in ascending order

    # Largest positive eigenvalue (last one if positive)
    pos_idx = -1  # Last eigenvalue
    pos_eigenvalue = eigenvalues[pos_idx]
    pos_eigenvector = eigenvectors[:, pos_idx]
    pos_eigenvector = pos_eigenvector / (pos_eigenvector.norm() + 1e-8)

    # Most negative eigenvalue (first one if negative)
    neg_idx = 0  # First eigenvalue
    neg_eigenvalue = eigenvalues[neg_idx]
    neg_eigenvector = eigenvectors[:, neg_idx]
    neg_eigenvector = neg_eigenvector / (neg_eigenvector.norm() + 1e-8)

    # Verify activations
    h_pos = (L_k @ pos_eigenvector) * (R_k @ pos_eigenvector)
    h_neg = (L_k @ neg_eigenvector) * (R_k @ neg_eigenvector)

    return {
        'positive_optimal': pos_eigenvector.cpu(),
        'positive_eigenvalue': pos_eigenvalue.item(),
        'positive_activation': h_pos.item(),
        'negative_optimal': neg_eigenvector.cpu(),
        'negative_eigenvalue': neg_eigenvalue.item(),
        'negative_activation': h_neg.item(),
        'L_k': L_k.cpu(),
        'R_k': R_k.cpu(),
        'L_R_alignment': F.cosine_similarity(L_k.unsqueeze(0), R_k.unsqueeze(0)).item(),
    }


def verify_layer0_optimal(channel, optimal_x):
    """Verify that optimal_x indeed gives maximum activation."""
    L_k = model.blocks[0].mlp.L.weight[channel].detach()
    R_k = model.blocks[0].mlp.R.weight[channel].detach()

    # Activation at optimal
    h_optimal = (L_k @ optimal_x.to(device)) * (R_k @ optimal_x.to(device))

    # Compare to random directions
    random_activations = []
    for _ in range(1000):
        x_rand = torch.randn(embed_dim, device=device)
        x_rand = x_rand / x_rand.norm()
        h_rand = (L_k @ x_rand) * (R_k @ x_rand)
        random_activations.append(h_rand.item())

    return {
        'optimal_activation': h_optimal.item(),
        'random_max': max(random_activations),
        'random_mean': np.mean(random_activations),
        'random_std': np.std(random_activations),
        'is_optimal': h_optimal.item() >= max(random_activations) * 0.99,
    }


# %%
# =============================================================================
# LAYER 1: GRADIENT-BASED OPTIMIZATION
# =============================================================================
#
# For Layer 1, the input is: RMSNorm(embed + D_0 @ h_0)
# where h_0 = L_0(embed) * R_0(embed)
#
# This is non-linear, so we optimize via gradient descent.

def optimize_layer1_input(channel, n_steps=500, lr=0.1, init='random'):
    """
    Find input embedding that maximizes Layer 1 channel activation.

    Args:
        channel: which Layer 1 channel to optimize for
        n_steps: optimization steps
        lr: learning rate
        init: 'random' or 'zeros'

    Returns:
        optimal_embed: [n_patches, embed_dim] - optimal patch embeddings
        activation_history: list of activations during optimization
    """
    # Initialize embedding (this is the input to the bilinear layers, after patch_embed)
    if init == 'random':
        embed = torch.randn(n_patches, embed_dim, device=device) * 0.1
    else:
        embed = torch.zeros(n_patches, embed_dim, device=device)

    embed = nn.Parameter(embed)
    optimizer = torch.optim.Adam([embed], lr=lr)

    activation_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward through Layer 0
        x = embed
        mlp_out_0, h_0 = model.blocks[0].mlp(x)
        residual = x + mlp_out_0
        x_normed = model.blocks[0].norm(residual)

        # Forward through Layer 1 (just compute h, not full output)
        h_1 = model.blocks[1].mlp.L(x_normed) * model.blocks[1].mlp.R(x_normed)

        # Target: maximize activation of specific channel (sum over patches)
        activation = h_1[:, channel].sum()

        # Maximize (so minimize negative)
        loss = -activation
        loss.backward()
        optimizer.step()

        # Normalize embedding to prevent explosion
        with torch.no_grad():
            embed.data = embed.data / (embed.data.norm(dim=-1, keepdim=True) + 1e-8)

        activation_history.append(activation.item())

    return embed.detach().cpu(), activation_history


def optimize_layer1_input_per_patch(channel, patch_idx, n_steps=500, lr=0.1):
    """
    Optimize input for a specific patch position in Layer 1.
    """
    embed = torch.randn(embed_dim, device=device) * 0.1
    embed = nn.Parameter(embed)
    optimizer = torch.optim.Adam([embed], lr=lr)

    activation_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Create full embedding with zeros except at target patch
        full_embed = torch.zeros(n_patches, embed_dim, device=device)
        full_embed[patch_idx] = embed

        # Forward through Layer 0
        mlp_out_0, h_0 = model.blocks[0].mlp(full_embed)
        residual = full_embed + mlp_out_0
        x_normed = model.blocks[0].norm(residual)

        # Forward through Layer 1
        h_1 = model.blocks[1].mlp.L(x_normed) * model.blocks[1].mlp.R(x_normed)

        activation = h_1[patch_idx, channel]
        loss = -activation
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            embed.data = embed.data / (embed.data.norm() + 1e-8)

        activation_history.append(activation.item())

    return embed.detach().cpu(), activation_history


# %%
# =============================================================================
# TOP ACTIVATING EXAMPLES
# =============================================================================

def get_all_activations(layer=0, n_samples=None):
    """
    Get activations for all test samples.

    Returns:
        all_h: [n_samples, n_patches, hidden_dim]
        all_labels: [n_samples]
    """
    if n_samples is None:
        n_samples = len(test_dataset)

    all_h = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc=f"Computing L{layer} activations"):
            x, y = test_dataset[idx]
            x_in = x.unsqueeze(0).to(device)

            # Forward through patch embed
            embed = model.patch_embed(x_in).squeeze(0)

            # Forward through layers
            residual = embed
            for li, block in enumerate(model.blocks):
                mlp_out, h = block.mlp(residual)

                if li == layer:
                    all_h.append(h.cpu())
                    break

                residual = residual + mlp_out
                residual = block.norm(residual)

            all_labels.append(y)

    all_h = torch.stack(all_h)  # [n_samples, n_patches, hidden_dim]
    all_labels = torch.tensor(all_labels)

    return all_h, all_labels


def find_top_activating(channel, layer=0, k=10, all_h=None, all_labels=None, n_samples=1000):
    """
    Find test images that most strongly activate a channel.

    Returns:
        top_indices: indices of top-k activating images
        top_activations: activation values
        top_patches: which patch had max activation
        top_labels: class labels
    """
    if all_h is None:
        all_h, all_labels = get_all_activations(layer=layer, n_samples=n_samples)

    # Get max activation per sample (across patches)
    channel_activations = all_h[:, :, channel]  # [n_samples, n_patches]
    max_per_sample, max_patch = channel_activations.max(dim=1)  # [n_samples]

    # Top k
    top_vals, top_idx = max_per_sample.topk(k)

    return {
        'indices': top_idx.numpy(),
        'activations': top_vals.numpy(),
        'patches': max_patch[top_idx].numpy(),
        'labels': all_labels[top_idx].numpy(),
    }


def find_top_activating_signed(channel, layer=0, k=10, all_h=None, all_labels=None, n_samples=1000):
    """
    Find images with most positive AND most negative activations.
    """
    if all_h is None:
        all_h, all_labels = get_all_activations(layer=layer, n_samples=n_samples)

    channel_activations = all_h[:, :, channel]

    # Most positive
    max_per_sample, max_patch = channel_activations.max(dim=1)
    top_pos_vals, top_pos_idx = max_per_sample.topk(k)

    # Most negative
    min_per_sample, min_patch = channel_activations.min(dim=1)
    top_neg_vals, top_neg_idx = min_per_sample.topk(k, largest=False)

    return {
        'positive': {
            'indices': top_pos_idx.numpy(),
            'activations': top_pos_vals.numpy(),
            'patches': max_patch[top_pos_idx].numpy(),
            'labels': all_labels[top_pos_idx].numpy(),
        },
        'negative': {
            'indices': top_neg_idx.numpy(),
            'activations': top_neg_vals.numpy(),
            'patches': min_patch[top_neg_idx].numpy(),
            'labels': all_labels[top_neg_idx].numpy(),
        }
    }


# %%
# =============================================================================
# VISUALIZATION
# =============================================================================

def show_image(x, title=None, ax=None):
    """Display a CIFAR image."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    img = x.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    if ax is None:
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()
    else:
        ax.imshow(img)
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=9)


def embed_to_patch(embed_vector, normalize_to_01=True):
    """
    Convert an embedding vector back to a 4x4 RGB patch.

    The patch embedding does: proj(raw_patch) + pos_embed = embed
    So: raw_patch ≈ proj^{-1}(embed - pos_embed)

    Since embed_dim = 3*4*4 = 48, proj is square and (usually) invertible.

    Args:
        embed_vector: [embed_dim] embedding in projected space
        normalize_to_01: if True, normalize patch values to [0, 1] range for display

    Returns:
        patch: [3, 4, 4] RGB patch image
    """
    # Get projection weights
    W_proj = model.patch_embed.proj.weight.detach()  # [embed_dim, 48]

    embed_vector = embed_vector.to(W_proj.device)

    # Invert projection: raw = W^{-1} @ embed
    # Use pseudoinverse for numerical stability
    W_pinv = torch.linalg.pinv(W_proj)  # [48, embed_dim]
    raw_patch = W_pinv @ embed_vector  # [48]

    # Reshape to [3, 4, 4]
    patch = raw_patch.reshape(3, 4, 4)

    if normalize_to_01:
        # Normalize to [0, 1] range for visualization
        # Center and scale by the overall range
        pmin, pmax = patch.min(), patch.max()
        if pmax > pmin:
            patch = (patch - pmin) / (pmax - pmin)
        else:
            patch = torch.ones_like(patch) * 0.5

    return patch.cpu()


def visualize_optimal_as_patch(channel):
    """
    Visualize optimal inputs for a Layer 0 channel as 4x4 RGB patches.

    Shows:
    - RGB patches (normalized to [0,1] for display)
    - Per-channel (R, G, B) heatmaps showing the raw pattern
    """
    result = get_layer0_optimal_input(channel)
    pos_x = result['positive_optimal']
    neg_x = result['negative_optimal']

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Get patches - normalized for RGB display
    pos_patch = embed_to_patch(pos_x, normalize_to_01=True)
    neg_patch = embed_to_patch(neg_x, normalize_to_01=True)

    # Get raw patches (not normalized) for heatmaps
    pos_patch_raw = embed_to_patch(pos_x, normalize_to_01=False)
    neg_patch_raw = embed_to_patch(neg_x, normalize_to_01=False)

    # Row 0: Positive optimal
    # RGB image
    ax = axes[0, 0]
    ax.imshow(pos_patch.permute(1, 2, 0).numpy(), interpolation='nearest')
    ax.set_title(f'Positive optimal\n(h={result["positive_activation"]:.3f})')
    ax.axis('off')

    # R, G, B channels as heatmaps
    vmax = max(pos_patch_raw.abs().max(), neg_patch_raw.abs().max())
    for c, color_name in enumerate(['Red', 'Green', 'Blue']):
        ax = axes[0, c + 1]
        im = ax.imshow(pos_patch_raw[c].numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(f'{color_name}')
        ax.axis('off')

    # Grayscale (mean of channels)
    ax = axes[0, 4]
    gray = pos_patch_raw.mean(dim=0).numpy()
    im = ax.imshow(gray, cmap='RdBu_r', vmin=-vmax/3, vmax=vmax/3, interpolation='nearest')
    ax.set_title('Gray (mean)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 1: Negative optimal
    ax = axes[1, 0]
    ax.imshow(neg_patch.permute(1, 2, 0).numpy(), interpolation='nearest')
    ax.set_title(f'Negative optimal\n(h={result["negative_activation"]:.3f})')
    ax.axis('off')

    for c, color_name in enumerate(['Red', 'Green', 'Blue']):
        ax = axes[1, c + 1]
        im = ax.imshow(neg_patch_raw[c].numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(f'{color_name}')
        ax.axis('off')

    ax = axes[1, 4]
    gray = neg_patch_raw.mean(dim=0).numpy()
    im = ax.imshow(gray, cmap='RdBu_r', vmin=-vmax/3, vmax=vmax/3, interpolation='nearest')
    ax.set_title('Gray (mean)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'Layer 0, Channel {channel}: Optimal Patches (4x4 pixels)', fontsize=12)
    plt.tight_layout()
    plt.show()

    return result


def visualize_optimal_input_layer0(channel):
    """
    Visualize both optimal input directions for a Layer 0 channel.
    Shows positive optimal (max activation) and negative optimal (min activation).
    """
    result = get_layer0_optimal_input(channel)
    pos_x = result['positive_optimal']
    neg_x = result['negative_optimal']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: Positive optimal
    ax = axes[0, 0]
    ax.bar(range(embed_dim), pos_x.numpy(), color='red', alpha=0.7)
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Value')
    ax.set_title(f'Positive Optimal Input\n(eigenvalue: {result["positive_eigenvalue"]:.3f}, h={result["positive_activation"]:.3f})')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    # Row 2: Negative optimal
    ax = axes[1, 0]
    ax.bar(range(embed_dim), neg_x.numpy(), color='blue', alpha=0.7)
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Value')
    ax.set_title(f'Negative Optimal Input\n(eigenvalue: {result["negative_eigenvalue"]:.3f}, h={result["negative_activation"]:.3f})')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    # L_k and R_k (same for both rows)
    for row in range(2):
        ax = axes[row, 1]
        ax.bar(range(embed_dim), result['L_k'].numpy(), alpha=0.5, label='L_k')
        ax.bar(range(embed_dim), result['R_k'].numpy(), alpha=0.5, label='R_k')
        ax.set_xlabel('Embedding dimension')
        ax.set_ylabel('Weight')
        ax.set_title(f'L and R weights (alignment: {result["L_R_alignment"]:.3f})')
        ax.legend()

    # Alignment comparison
    pos_L_sim = F.cosine_similarity(pos_x.unsqueeze(0), result['L_k'].unsqueeze(0)).item()
    pos_R_sim = F.cosine_similarity(pos_x.unsqueeze(0), result['R_k'].unsqueeze(0)).item()
    neg_L_sim = F.cosine_similarity(neg_x.unsqueeze(0), result['L_k'].unsqueeze(0)).item()
    neg_R_sim = F.cosine_similarity(neg_x.unsqueeze(0), result['R_k'].unsqueeze(0)).item()
    pos_neg_sim = F.cosine_similarity(pos_x.unsqueeze(0), neg_x.unsqueeze(0)).item()

    ax = axes[0, 2]
    ax.bar(['cos(pos,L)', 'cos(pos,R)', 'L-R align'], [pos_L_sim, pos_R_sim, result['L_R_alignment']], color='red', alpha=0.7)
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Positive optimal alignment')
    ax.set_ylim(-1, 1)

    ax = axes[1, 2]
    ax.bar(['cos(neg,L)', 'cos(neg,R)', 'cos(pos,neg)'], [neg_L_sim, neg_R_sim, pos_neg_sim], color='blue', alpha=0.7)
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Negative optimal alignment')
    ax.set_ylim(-1, 1)

    plt.suptitle(f'Layer 0, Channel {channel}: Optimal Inputs', fontsize=14)
    plt.tight_layout()
    plt.show()

    return result


def visualize_optimal_input_layer1(channel, optimal_embed, activation_history):
    """
    Visualize optimized input for a Layer 1 channel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Optimization curve
    ax = axes[0]
    ax.plot(activation_history)
    ax.set_xlabel('Step')
    ax.set_ylabel('Activation')
    ax.set_title(f'Channel {channel} Optimization\n(final: {activation_history[-1]:.3f})')

    # Embedding magnitude per patch (8x8)
    ax = axes[1]
    embed_norms = optimal_embed.norm(dim=-1).reshape(8, 8).numpy()
    im = ax.imshow(embed_norms, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title('Embedding magnitude per patch')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Mean embedding direction
    ax = axes[2]
    mean_embed = optimal_embed.mean(dim=0).numpy()
    ax.bar(range(embed_dim), mean_embed)
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Value')
    ax.set_title('Mean optimal embedding')

    plt.tight_layout()
    plt.show()


def show_top_activating(channel, layer=0, k=5, all_h=None, all_labels=None):
    """
    Display top activating images for a channel.
    """
    results = find_top_activating_signed(channel, layer, k=k, all_h=all_h, all_labels=all_labels)

    fig, axes = plt.subplots(2, k, figsize=(3*k, 6))

    # Top positive
    for i, idx in enumerate(results['positive']['indices']):
        x, y = test_dataset[idx]
        ax = axes[0, i]
        show_image(x, ax=ax)
        patch = results['positive']['patches'][i]
        patch_y, patch_x = patch // 8, patch % 8
        ax.set_title(f"{CLASSES[y]}\nh={results['positive']['activations'][i]:.2f}\npatch=({patch_y},{patch_x})", fontsize=8)

    axes[0, 0].set_ylabel('Top Positive', fontsize=12)

    # Top negative
    for i, idx in enumerate(results['negative']['indices']):
        x, y = test_dataset[idx]
        ax = axes[1, i]
        show_image(x, ax=ax)
        patch = results['negative']['patches'][i]
        patch_y, patch_x = patch // 8, patch % 8
        ax.set_title(f"{CLASSES[y]}\nh={results['negative']['activations'][i]:.2f}\npatch=({patch_y},{patch_x})", fontsize=8)

    axes[1, 0].set_ylabel('Top Negative', fontsize=12)

    plt.suptitle(f'Layer {layer}, Channel {channel}: Top Activating Images', fontsize=14)
    plt.tight_layout()
    plt.show()

    return results


# %%
# =============================================================================
# FULL ANALYSIS WORKFLOW
# =============================================================================

def analyze_channel_layer0(channel):
    """Full analysis for a Layer 0 channel."""
    print(f"\n{'='*60}")
    print(f"LAYER 0, CHANNEL {channel}")
    print(f"{'='*60}")

    # Analytical optimal
    result = get_layer0_optimal_input(channel)
    print(f"\nOptimal input analysis:")
    print(f"  Positive optimal: eigenvalue={result['positive_eigenvalue']:.4f}, activation={result['positive_activation']:.4f}")
    print(f"  Negative optimal: eigenvalue={result['negative_eigenvalue']:.4f}, activation={result['negative_activation']:.4f}")
    print(f"  L-R alignment (cosine sim): {result['L_R_alignment']:.4f}")

    # Verify both
    verify_pos = verify_layer0_optimal(channel, result['positive_optimal'])
    verify_neg = verify_layer0_optimal(channel, result['negative_optimal'])
    print(f"\nVerification (vs 1000 random):")
    print(f"  Positive optimal activation: {verify_pos['optimal_activation']:.4f}")
    print(f"  Negative optimal activation: {verify_neg['optimal_activation']:.4f}")
    print(f"  Random max: {verify_pos['random_max']:.4f}")
    print(f"  Random min: {min(verify_pos['random_mean'] - 2*verify_pos['random_std'], verify_neg['optimal_activation']):.4f}")

    # Visualize optimal as patches (the main visualization!)
    visualize_optimal_as_patch(channel)

    # Visualize optimal (embedding space details)
    visualize_optimal_input_layer0(channel)

    # Top activating examples
    print("\nTop activating examples:")
    show_top_activating(channel, layer=0, k=5)

    return result


def analyze_channel_layer1(channel, n_steps=500):
    """Full analysis for a Layer 1 channel."""
    print(f"\n{'='*60}")
    print(f"LAYER 1, CHANNEL {channel}")
    print(f"{'='*60}")

    # Optimize input
    print("\nOptimizing input...")
    optimal_embed, history = optimize_layer1_input(channel, n_steps=n_steps)
    print(f"  Final activation: {history[-1]:.4f}")

    # Visualize
    visualize_optimal_input_layer1(channel, optimal_embed, history)

    # Top activating examples
    print("\nTop activating examples:")
    show_top_activating(channel, layer=1, k=5)

    return optimal_embed, history


# %%
# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def compute_all_layer0_stats():
    """Compute statistics for all Layer 0 channels."""
    pos_optimal, neg_optimal, pos_eig, neg_eig, alignments = compute_layer0_optimal_inputs()

    print("\nLayer 0 Channel Statistics:")
    print(f"  Positive eigenvalue range: [{pos_eig.min():.3f}, {pos_eig.max():.3f}]")
    print(f"  Negative eigenvalue range: [{neg_eig.min():.3f}, {neg_eig.max():.3f}]")
    print(f"  L-R alignment range: [{alignments.min():.3f}, {alignments.max():.3f}]")
    print(f"  Mean L-R alignment: {alignments.mean():.3f}")

    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(pos_eig.numpy(), bins=30, alpha=0.7, label='Positive (max)', color='red')
    ax.hist(neg_eig.numpy(), bins=30, alpha=0.7, label='Negative (min)', color='blue')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of eigenvalues')
    ax.legend()

    ax = axes[1]
    ax.hist(alignments.numpy(), bins=30)
    ax.set_xlabel('L-R alignment (cosine sim)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of L-R alignments')

    ax = axes[2]
    ax.scatter(alignments.numpy(), pos_eig.numpy(), alpha=0.5, label='Positive', color='red')
    ax.scatter(alignments.numpy(), neg_eig.numpy(), alpha=0.5, label='Negative', color='blue')
    ax.set_xlabel('L-R alignment')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Alignment vs Eigenvalue')
    ax.legend()
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pos_optimal, neg_optimal, pos_eig, neg_eig, alignments


# %%
# =============================================================================
# EXAMPLES
# =============================================================================

# %%
# EXAMPLE 1: Visualize optimal input as a 4x4 patch (what the channel "wants to see")
visualize_optimal_as_patch(channel=0)

# %%
# EXAMPLE 2: Full analysis for a single Layer 0 channel (patches + embeddings + top activating)
analyze_channel_layer0(channel=0)

# %%
# EXAMPLE 3: Analyze a single Layer 1 channel
analyze_channel_layer1(channel=0, n_steps=300)

# %%
# EXAMPLE 4: Compute all Layer 0 optimal inputs and statistics
pos_optimal, neg_optimal, pos_eig, neg_eig, alignments = compute_all_layer0_stats()

# %%
# EXAMPLE 5: Find channels with highest/lowest L-R alignment
print("Channels with highest L-R alignment (L and R point same direction):")
top_aligned = alignments.argsort(descending=True)[:5]
for ch in top_aligned:
    print(f"  Channel {ch}: alignment={alignments[ch]:.3f}, pos_eig={pos_eig[ch]:.3f}, neg_eig={neg_eig[ch]:.3f}")

print("\nChannels with lowest L-R alignment (L and R point opposite directions):")
bot_aligned = alignments.argsort()[:5]
for ch in bot_aligned:
    print(f"  Channel {ch}: alignment={alignments[ch]:.3f}, pos_eig={pos_eig[ch]:.3f}, neg_eig={neg_eig[ch]:.3f}")

# %%
# EXAMPLE 6: Compare optimal vs real activations for a channel
channel = 0
result = get_layer0_optimal_input(channel)
print(f"\nChannel {channel}:")
print(f"  Theoretical positive max: {result['positive_activation']:.4f}")
print(f"  Theoretical negative max: {result['negative_activation']:.4f}")

# Get real activations from dataset
all_h, all_labels = get_all_activations(layer=0, n_samples=1000)
real_max = all_h[:, :, channel].max().item()
real_min = all_h[:, :, channel].min().item()
real_mean = all_h[:, :, channel].mean().item()
print(f"  Real max activation (1000 samples): {real_max:.4f}")
print(f"  Real min activation (1000 samples): {real_min:.4f}")
print(f"  Real mean activation: {real_mean:.4f}")
print(f"  Ratio (real_max / theoretical_pos): {real_max / result['positive_activation']:.3f}")
print(f"  Ratio (real_min / theoretical_neg): {real_min / result['negative_activation']:.3f}")

# %%
# EXAMPLE 7: Show top activating for multiple channels
for ch in [0, 10, 50, 100]:
    print(f"\n--- Channel {ch} ---")
    show_top_activating(ch, layer=0, k=3)

# %%
# EXAMPLE 8: Layer 1 optimization for multiple channels
for ch in [0, 50]:
    print(f"\n--- Layer 1, Channel {ch} ---")
    opt_embed, history = optimize_layer1_input(ch, n_steps=200)
    print(f"  Final activation: {history[-1]:.3f}")
    visualize_optimal_input_layer1(ch, opt_embed, history)

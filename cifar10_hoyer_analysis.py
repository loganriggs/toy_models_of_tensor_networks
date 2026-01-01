# %%
"""
CIFAR-10 Hoyer Sparsity Analysis

Analyzes the trained sparse bilinear model:
1. Channel activation frequency across test set
2. Direct logit composition (D @ head)
3. Downstream composition (D @ L, D @ R for later blocks)
4. Per-sample analysis for correctly classified examples
5. Channel activation correlations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# %%
# Model definition (same as training)

class BilinearResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, bias=False):
        super().__init__()
        self.L = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.R = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.D = nn.Linear(hidden_dim, in_dim, bias=bias)

    def forward(self, x):
        u = self.L(x)
        v = self.R(x)
        h = u * v
        return x + self.D(h), h


class BilinearNet(nn.Module):
    def __init__(self, in_dim=3072, embed_dim=None, hidden_dim=64, n_blocks=1, n_classes=10):
        super().__init__()
        if embed_dim is not None:
            self.embed = nn.Linear(in_dim, embed_dim, bias=False)
            working_dim = embed_dim
        else:
            self.embed = None
            working_dim = in_dim

        self.blocks = nn.ModuleList([
            BilinearResBlock(working_dim, hidden_dim) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(working_dim, n_classes, bias=False)
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(1)
        if self.embed is not None:
            x = self.embed(x)
        all_h = []
        for block in self.blocks:
            x, h = block(x)
            all_h.append(h)
        return self.head(x), all_h

# %%
# Load model and data

ckpt = torch.load('cifar10_4blocks_h16_hoyer5_warmup.pt', map_location=DEVICE, weights_only=False)
print(f"Loaded checkpoint:")
print(f"  Lambda: {ckpt['lambda_max']}")
print(f"  Hidden dim: {ckpt['hidden_dim']}")
print(f"  N blocks: {ckpt['n_blocks']}")
print(f"  Final acc: {ckpt['final_acc']:.2%}")
print(f"  Final sparsity: {1 - ckpt['final_hoyer']:.2%}")

model = BilinearNet(
    in_dim=3072,
    embed_dim=ckpt['embed_dim'],
    hidden_dim=ckpt['hidden_dim'],
    n_blocks=ckpt['n_blocks'],
    n_classes=10,
).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Data
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
test_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_data = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

n_blocks = model.n_blocks
hidden_dim = model.hidden_dim
total_channels = n_blocks * hidden_dim
print(f"\nTotal channels: {total_channels} ({n_blocks} blocks x {hidden_dim} channels)")

# %%
# (1) Collect channel activations across test set

@torch.no_grad()
def collect_activations(model, loader, device):
    """Collect all channel activations and predictions across dataset."""
    all_h = []
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, h_list = model(x)

        # Concatenate hidden activations: [B, total_channels]
        h_cat = torch.cat(h_list, dim=1)
        all_h.append(h_cat.cpu())
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())

    all_h = torch.cat(all_h, dim=0)  # [N, total_channels]
    all_preds = torch.cat(all_preds, dim=0)  # [N]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    return all_h, all_preds, all_labels

print("Collecting activations across test set...")
all_h, all_preds, all_labels = collect_activations(model, test_loader, DEVICE)
print(f"Activations shape: {all_h.shape}")
print(f"Test accuracy: {(all_preds == all_labels).float().mean():.2%}")

# %%
# (1) Plot channel activation frequency (histogram with log_x)

def plot_activation_frequency(all_h, threshold=0.01):
    """Plot histogram of channel activation frequencies."""
    # Count how often each channel is "active" (|h| > threshold)
    active = (all_h.abs() > threshold).float()
    freq_per_channel = active.mean(dim=0).numpy()  # [total_channels]

    # Also compute mean absolute activation per channel
    mean_abs = all_h.abs().mean(dim=0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Frequency histogram
    ax = axes[0]
    ax.hist(freq_per_channel, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Activation Frequency')
    ax.set_ylabel('Number of Channels')
    ax.set_title(f'Channel Activation Frequency (threshold={threshold})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Mean absolute activation histogram
    ax = axes[1]
    ax.hist(mean_abs, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Mean |activation|')
    ax.set_ylabel('Number of Channels')
    ax.set_title('Mean Absolute Activation per Channel')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cifar10_hoyer_activation_freq.png', dpi=150)
    plt.show()

    print(f"\nActivation frequency stats:")
    print(f"  Min: {freq_per_channel.min():.4f}")
    print(f"  Max: {freq_per_channel.max():.4f}")
    print(f"  Mean: {freq_per_channel.mean():.4f}")
    print(f"  Median: {np.median(freq_per_channel):.4f}")
    print(f"  Channels with freq < 0.01: {(freq_per_channel < 0.01).sum()}")
    print(f"  Channels with freq < 0.1: {(freq_per_channel < 0.1).sum()}")

    return freq_per_channel, mean_abs

freq_per_channel, mean_abs_per_channel = plot_activation_frequency(all_h, threshold=10)

# %%
# (2) Direct logit composition: head @ D for each channel

def compute_direct_logit_effects(model):
    """
    Compute direct effect of each hidden channel on each output class.

    For each block b and channel c:
        effect[b, c, class] = head.weight[class] @ D.weight[:, c]

    Returns:
        effects: [n_blocks, hidden_dim, n_classes]
    """
    head_W = model.head.weight.detach()  # [n_classes, embed_dim]

    effects = []
    for block in model.blocks:
        D_W = block.D.weight.detach()  # [embed_dim, hidden_dim]
        # effect[class, channel] = sum_i(head[class, i] * D[i, channel])
        effect = head_W @ D_W  # [n_classes, hidden_dim]
        effects.append(effect.cpu())

    effects = torch.stack(effects, dim=0)  # [n_blocks, n_classes, hidden_dim]
    effects = effects.permute(0, 2, 1)  # [n_blocks, hidden_dim, n_classes]

    return effects

direct_effects = compute_direct_logit_effects(model)
print(f"Direct effects shape: {direct_effects.shape}")  # [n_blocks, hidden_dim, n_classes]

# Flatten for easier indexing: [total_channels, n_classes]
direct_effects_flat = direct_effects.reshape(-1, 10)
print(f"Direct effects flat shape: {direct_effects_flat.shape}")

# %%
# (3) Downstream composition: D @ L and D @ R for later blocks

def compute_downstream_effects(model):
    """
    Compute how each channel's D output affects downstream L and R inputs.

    For block i, channel c, and downstream block j > i:
        effect_L[i, c, j, c'] = L_j.weight[c', :] @ D_i.weight[:, c]
        effect_R[i, c, j, c'] = R_j.weight[c', :] @ D_i.weight[:, c]

    Returns:
        downstream_L: dict of (src_block, dst_block) -> [hidden_dim, hidden_dim]
        downstream_R: dict of (src_block, dst_block) -> [hidden_dim, hidden_dim]
    """
    n_blocks = len(model.blocks)
    downstream_L = {}
    downstream_R = {}

    for i in range(n_blocks):
        D_i = model.blocks[i].D.weight.detach()  # [embed_dim, hidden_dim]

        for j in range(i + 1, n_blocks):
            L_j = model.blocks[j].L.weight.detach()  # [hidden_dim, embed_dim]
            R_j = model.blocks[j].R.weight.detach()  # [hidden_dim, embed_dim]

            # effect[dst_channel, src_channel] = L_j @ D_i
            effect_L = L_j @ D_i  # [hidden_dim, hidden_dim]
            effect_R = R_j @ D_i  # [hidden_dim, hidden_dim]

            downstream_L[(i, j)] = effect_L.cpu()
            downstream_R[(i, j)] = effect_R.cpu()

    return downstream_L, downstream_R

downstream_L, downstream_R = compute_downstream_effects(model)
print(f"Downstream connections computed for {len(downstream_L)} block pairs")

# %%
# Visualize direct logit effects

def plot_direct_effects(direct_effects, class_names):
    """Plot heatmap of direct logit effects."""
    n_blocks, hidden_dim, n_classes = direct_effects.shape

    fig, axes = plt.subplots(1, n_blocks, figsize=(4 * n_blocks, 6))
    if n_blocks == 1:
        axes = [axes]

    for b in range(n_blocks):
        ax = axes[b]
        effect = direct_effects[b].numpy()  # [hidden_dim, n_classes]

        im = ax.imshow(effect, aspect='auto', cmap='RdBu_r',
                       vmin=-effect.max(), vmax=effect.max())
        ax.set_xlabel('Class')
        ax.set_ylabel('Channel')
        ax.set_title(f'Block {b}: D @ head')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('cifar10_hoyer_direct_effects.png', dpi=150)
    plt.show()

    return

plot_direct_effects(direct_effects, CIFAR_CLASSES)

# %%
# (4) Per-sample analysis for a correctly classified example

def analyze_single_sample(model, x, y, direct_effects_flat, downstream_L, downstream_R,
                          class_names, device):
    """
    Analyze a single correctly classified sample.

    Args:
        x: input image [1, 3, 32, 32]
        y: true label (int)
        direct_effects_flat: [total_channels, n_classes]
        downstream_L, downstream_R: dicts from compute_downstream_effects
        class_names: list of class names

    Returns:
        dict with analysis results
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits, h_list = model(x)
        pred = logits.argmax(1).item()

    h_cat = torch.cat(h_list, dim=1).squeeze(0).cpu()  # [total_channels]
    logits = logits.squeeze(0).cpu()

    # Per-channel contribution to each class
    # contribution[channel, class] = h[channel] * direct_effect[channel, class]
    contributions = h_cat.unsqueeze(1) * direct_effects_flat  # [total_channels, n_classes]

    # Contribution to predicted class
    contrib_to_pred = contributions[:, pred].numpy()

    # Contribution to true class
    contrib_to_true = contributions[:, y].numpy()

    # Downstream effects weighted by activation
    n_blocks = len(h_list)
    hidden_dim = h_list[0].shape[1]

    downstream_contrib = {}
    for (i, j), effect_L in downstream_L.items():
        effect_R = downstream_R[(i, j)]
        h_i = h_list[i].squeeze(0).cpu()  # [hidden_dim]

        # Weighted effect on downstream channels
        weighted_L = effect_L * h_i.unsqueeze(0)  # [hidden_dim_dst, hidden_dim_src]
        weighted_R = effect_R * h_i.unsqueeze(0)

        downstream_contrib[(i, j)] = {
            'L': weighted_L.sum(dim=1).numpy(),  # [hidden_dim_dst]
            'R': weighted_R.sum(dim=1).numpy(),
        }

    return {
        'h': h_cat.numpy(),
        'logits': logits.numpy(),
        'pred': pred,
        'true': y,
        'contributions': contributions.numpy(),
        'contrib_to_pred': contrib_to_pred,
        'contrib_to_true': contrib_to_true,
        'downstream_contrib': downstream_contrib,
        'h_list': [h.squeeze(0).cpu().numpy() for h in h_list],
    }

# Find a correctly classified sample
correct_mask = (all_preds == all_labels)
correct_indices = torch.where(correct_mask)[0]
print(f"Correctly classified: {len(correct_indices)} / {len(all_labels)}")

# Pick a sample (e.g., first correct one)
sample_idx = correct_indices[0].item()
sample_x, sample_y = test_data[sample_idx]
sample_x = sample_x.unsqueeze(0)  # Add batch dim

sample_analysis = analyze_single_sample(
    model, sample_x, sample_y, direct_effects_flat,
    downstream_L, downstream_R, CIFAR_CLASSES, DEVICE
)

print(f"\nSample {sample_idx}:")
print(f"  True class: {CIFAR_CLASSES[sample_analysis['true']]}")
print(f"  Predicted: {CIFAR_CLASSES[sample_analysis['pred']]}")
print(f"  Active channels (|h| > 0.1): {(np.abs(sample_analysis['h']) > 0.1).sum()}")

# %%
# Plot per-sample contributions (2 & 3)

def plot_sample_analysis(sample_analysis, class_names, n_blocks, hidden_dim):
    """Plot contribution analysis for a single sample."""
    h = sample_analysis['h']
    contrib_to_pred = sample_analysis['contrib_to_pred']
    contrib_to_true = sample_analysis['contrib_to_true']
    downstream = sample_analysis['downstream_contrib']
    pred = sample_analysis['pred']
    true_label = sample_analysis['true']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Channel activations by block
    ax = axes[0, 0]
    for b in range(n_blocks):
        start = b * hidden_dim
        end = (b + 1) * hidden_dim
        h_block = h[start:end]
        x_pos = np.arange(hidden_dim) + b * (hidden_dim + 2)
        ax.bar(x_pos, h_block, label=f'Block {b}', alpha=0.7)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Activation')
    ax.set_title(f'Channel Activations (True: {class_names[true_label]}, Pred: {class_names[pred]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Direct contribution to predicted class
    ax = axes[0, 1]
    colors = ['C0' if c >= 0 else 'C3' for c in contrib_to_pred]
    for b in range(n_blocks):
        start = b * hidden_dim
        end = (b + 1) * hidden_dim
        x_pos = np.arange(hidden_dim) + b * (hidden_dim + 2)
        ax.bar(x_pos, contrib_to_pred[start:end],
               color=[colors[i] for i in range(start, end)], alpha=0.7)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Contribution to logit')
    ax.set_title(f'Direct Contribution to Predicted Class ({class_names[pred]})')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 3: Downstream effects (block 0 -> later blocks)
    ax = axes[1, 0]
    if len(downstream) > 0:
        # Show effect of block 0 on block 1's L and R
        if (0, 1) in downstream:
            effect_L = downstream[(0, 1)]['L']
            effect_R = downstream[(0, 1)]['R']
            x_pos = np.arange(hidden_dim)
            width = 0.35
            ax.bar(x_pos - width/2, effect_L, width, label='-> L₁', alpha=0.7)
            ax.bar(x_pos + width/2, effect_R, width, label='-> R₁', alpha=0.7)
            ax.set_xlabel('Downstream Channel (Block 1)')
            ax.set_ylabel('Weighted Effect')
            ax.set_title('Block 0 -> Block 1 Downstream Effect')
            ax.legend()
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No downstream blocks', ha='center', va='center')

    # Plot 4: All downstream effects summary
    ax = axes[1, 1]
    if len(downstream) > 0:
        all_effects = []
        labels = []
        for (i, j) in sorted(downstream.keys()):
            all_effects.append(np.abs(downstream[(i, j)]['L']).sum() +
                              np.abs(downstream[(i, j)]['R']).sum())
            labels.append(f'{i}→{j}')
        ax.bar(labels, all_effects, alpha=0.7)
        ax.set_xlabel('Block Connection')
        ax.set_ylabel('Total |Effect|')
        ax.set_title('Downstream Effect Magnitude by Connection')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cifar10_hoyer_sample_analysis.png', dpi=150)
    plt.show()

plot_sample_analysis(sample_analysis, CIFAR_CLASSES, n_blocks, hidden_dim)

# %%
# (5) Channel activation correlations

def compute_activation_correlations(all_h):
    """Compute correlation matrix of channel activations."""
    # all_h: [N, total_channels]
    h_np = all_h.numpy()

    # Compute correlation matrix
    corr = np.corrcoef(h_np.T)  # [total_channels, total_channels]

    return corr

print("Computing activation correlations...")
corr_matrix = compute_activation_correlations(all_h)
print(f"Correlation matrix shape: {corr_matrix.shape}")

# %%
# Plot correlation matrix

def plot_correlation_matrix(corr_matrix, n_blocks, hidden_dim):
    """Plot channel activation correlation matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add block boundaries
    for b in range(1, n_blocks):
        pos = b * hidden_dim - 0.5
        ax.axhline(y=pos, color='black', linewidth=2)
        ax.axvline(x=pos, color='black', linewidth=2)

    # Labels
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
    ax.set_title('Channel Activation Correlations')

    # Add block labels
    for b in range(n_blocks):
        mid = (b + 0.5) * hidden_dim
        ax.text(mid, -2, f'Block {b}', ha='center', fontsize=10)
        ax.text(-2, mid, f'Block {b}', ha='right', va='center', fontsize=10, rotation=90)

    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()
    plt.savefig('cifar10_hoyer_correlations.png', dpi=150)
    plt.show()

    # Stats
    # Get off-diagonal correlations
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    off_diag = corr_matrix[mask]

    print(f"\nCorrelation stats (off-diagonal):")
    print(f"  Mean: {off_diag.mean():.4f}")
    print(f"  Std: {off_diag.std():.4f}")
    print(f"  Min: {off_diag.min():.4f}")
    print(f"  Max: {off_diag.max():.4f}")
    print(f"  |corr| > 0.5: {(np.abs(off_diag) > 0.5).sum()}")
    print(f"  |corr| > 0.3: {(np.abs(off_diag) > 0.3).sum()}")

    return

plot_correlation_matrix(corr_matrix, n_blocks, hidden_dim)

# %%
# Within-block vs across-block correlations

def analyze_block_correlations(corr_matrix, n_blocks, hidden_dim):
    """Analyze correlations within vs across blocks."""
    within_block = []
    across_block = []

    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            block_i = i // hidden_dim
            block_j = j // hidden_dim

            if block_i == block_j:
                within_block.append(corr_matrix[i, j])
            else:
                across_block.append(corr_matrix[i, j])

    within_block = np.array(within_block)
    across_block = np.array(across_block)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(within_block, bins=50, alpha=0.7, label=f'Within-block (n={len(within_block)})', density=True)
    ax.hist(across_block, bins=50, alpha=0.7, label=f'Across-block (n={len(across_block)})', density=True)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Within-block vs Across-block Correlations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cifar10_hoyer_corr_within_across.png', dpi=150)
    plt.show()

    print(f"\nWithin-block correlations:")
    print(f"  Mean: {within_block.mean():.4f}")
    print(f"  Std: {within_block.std():.4f}")

    print(f"\nAcross-block correlations:")
    print(f"  Mean: {across_block.mean():.4f}")
    print(f"  Std: {across_block.std():.4f}")

    return within_block, across_block

within_corr, across_corr = analyze_block_correlations(corr_matrix, n_blocks, hidden_dim)

# %%
# Summary of all saved plots
print("\n" + "="*60)
print("SAVED PLOTS:")
print("="*60)
print("  cifar10_hoyer_activation_freq.png - Channel activation frequency")
print("  cifar10_hoyer_direct_effects.png - Direct logit effects (D @ head)")
print("  cifar10_hoyer_sample_analysis.png - Per-sample analysis")
print("  cifar10_hoyer_correlations.png - Full correlation matrix")
print("  cifar10_hoyer_corr_within_across.png - Within vs across block correlations")

# %%
# Expose key variables for inspection
print("\n" + "="*60)
print("KEY VARIABLES FOR INSPECTION:")
print("="*60)
print("  model - the loaded BilinearNet")
print("  all_h - all activations [N, total_channels]")
print("  all_preds, all_labels - predictions and labels")
print("  direct_effects - [n_blocks, hidden_dim, n_classes]")
print("  direct_effects_flat - [total_channels, n_classes]")
print("  downstream_L, downstream_R - dicts of downstream effects")
print("  sample_analysis - dict with per-sample analysis")
print("  corr_matrix - [total_channels, total_channels] correlations")
print("  freq_per_channel - activation frequency per channel")

# %%
# Plot activation histograms for 5 random channels
np.random.seed(42)
random_channels = np.random.choice(all_h.shape[1], 5, replace=False)
random_channels.sort()

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.tab10(np.linspace(0, 1, 5))
for i, ch in enumerate(random_channels):
    block_idx = ch // hidden_dim
    ch_in_block = ch % hidden_dim
    activations = all_h[:, ch].numpy()
    ax.hist(activations, bins=50, alpha=0.5, color=colors[i],
            label=f'Ch {ch} (Block {block_idx}, ch {ch_in_block})')

ax.set_xlabel('Activation Value', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Activation Distribution for 5 Random Channels', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cifar10_hoyer_channel_histograms.png', dpi=150)
print("Saved: cifar10_hoyer_channel_histograms.png")
plt.show()

# %%
# D-vector norms per layer
# Each D matrix is [embed_dim, hidden_dim], so columns are the d-vectors for each channel

def compute_d_vector_norms(model):
    """Compute L2 norms of D-vector columns for each block."""
    d_norms_per_block = []
    for block in model.blocks:
        D_W = block.D.weight.detach()  # [embed_dim, hidden_dim]
        # Column norms (each column is a d-vector)
        norms = D_W.norm(dim=0).cpu().numpy()  # [hidden_dim]
        d_norms_per_block.append(norms)
    return d_norms_per_block

d_norms = compute_d_vector_norms(model)

# Plot histograms overlaid by layer
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.tab10(np.linspace(0, 0.4, n_blocks))
for b, norms in enumerate(d_norms):
    ax.hist(norms, bins=15, alpha=0.5, color=colors[b],
            label=f'Block {b} (mean={norms.mean():.3f})')

ax.set_xlabel('D-vector L2 Norm', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of D-vector Norms by Layer', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cifar10_hoyer_d_norms.png', dpi=150)
print("Saved: cifar10_hoyer_d_norms.png")
plt.show()

# Print stats
print("\nD-vector norm stats by block:")
for b, norms in enumerate(d_norms):
    print(f"  Block {b}: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}, std={norms.std():.4f}")

# %%
# Cosine similarity of head output vectors (class embeddings)
# head.weight is [n_classes, embed_dim], each row is a class vector

head_W = model.head.weight.detach().cpu()  # [10, embed_dim]
head_W_norm = head_W / head_W.norm(dim=1, keepdim=True)  # normalize rows
cos_sim = head_W_norm @ head_W_norm.T  # [10, 10]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cos_sim.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha='right')
ax.set_yticklabels(CIFAR_CLASSES)
ax.set_title('Cosine Similarity of Head Output Vectors', fontsize=14)

# Add values in cells
for i in range(10):
    for j in range(10):
        val = cos_sim[i, j].item()
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

plt.colorbar(im, ax=ax, label='Cosine Similarity')
plt.tight_layout()
plt.savefig('cifar10_hoyer_head_cossim.png', dpi=150)
print("Saved: cifar10_hoyer_head_cossim.png")
plt.show()

# Print some stats
off_diag_mask = ~torch.eye(10, dtype=bool)
off_diag = cos_sim[off_diag_mask]
print(f"\nHead cosine similarity stats (off-diagonal):")
print(f"  Mean: {off_diag.mean():.4f}")
print(f"  Std: {off_diag.std():.4f}")
print(f"  Min: {off_diag.min():.4f}")
print(f"  Max: {off_diag.max():.4f}")

# %%
# Case study: Ablation by zeroing channels below threshold
# For a correctly classified sample, sweep threshold and track logits/accuracy

class BilinearNetWithAblation(nn.Module):
    """BilinearNet that can zero out channels below a threshold during forward."""
    def __init__(self, base_model, threshold=0.0):
        super().__init__()
        self.embed = base_model.embed
        self.blocks = base_model.blocks
        self.head = base_model.head
        self.threshold = threshold

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(1)
        if self.embed is not None:
            x = self.embed(x)

        all_h = []
        n_zeroed = 0
        n_total = 0

        for block in self.blocks:
            u = block.L(x)
            v = block.R(x)
            h = u * v

            # Zero out channels below threshold
            mask = h.abs() >= self.threshold
            h_masked = h * mask.float()
            n_zeroed += (~mask).sum().item()
            n_total += mask.numel()

            all_h.append(h_masked)
            x = x + block.D(h_masked)

        return self.head(x), all_h, n_zeroed, n_total

# Get a correctly classified sample
correct_mask = (all_preds == all_labels)
correct_indices = torch.where(correct_mask)[0]
sample_idx = correct_indices[0].item()

# Get the sample
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
test_data = datasets.CIFAR10('./data', train=False, download=False, transform=test_transform)
x_sample, y_sample = test_data[sample_idx]
x_sample = x_sample.unsqueeze(0).to(DEVICE)

print(f"Sample {sample_idx}: True class = {CIFAR_CLASSES[y_sample]}")

# Sweep thresholds
thresholds = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
results = []

ablation_model = BilinearNetWithAblation(model, threshold=0.0)
ablation_model.eval()

for thresh in thresholds:
    ablation_model.threshold = thresh
    with torch.no_grad():
        logits, h_list, n_zeroed, n_total = ablation_model(x_sample)

    probs = F.softmax(logits, dim=1)[0]
    pred_class = logits.argmax(1).item()
    correct_logit = logits[0, y_sample].item()
    correct_prob = probs[y_sample].item()
    pred_logit = logits[0, pred_class].item()
    frac_zeroed = n_zeroed / n_total

    results.append({
        'threshold': thresh,
        'correct_logit': correct_logit,
        'correct_prob': correct_prob,
        'pred_class': pred_class,
        'pred_logit': pred_logit,
        'is_correct': pred_class == y_sample,
        'frac_zeroed': frac_zeroed,
        'n_active': n_total - n_zeroed,
    })

    print(f"Thresh={thresh:4d}: {frac_zeroed*100:5.1f}% zeroed, "
          f"pred={CIFAR_CLASSES[pred_class]:10s}, "
          f"correct_logit={correct_logit:7.2f}, correct_prob={correct_prob:.3f}")

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Logit for correct class vs threshold
ax = axes[0]
ax.plot([r['threshold'] for r in results], [r['correct_logit'] for r in results], 'b-o', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Logit for Correct Class', fontsize=12)
ax.set_title(f'Correct Class ({CIFAR_CLASSES[y_sample]}) Logit vs Threshold', fontsize=12)
ax.set_xscale('symlog', linthresh=1)
ax.grid(True, alpha=0.3)

# Plot 2: Probability for correct class vs threshold
ax = axes[1]
ax.plot([r['threshold'] for r in results], [r['correct_prob'] for r in results], 'g-o', linewidth=2)
ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Random guess')
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Probability for Correct Class', fontsize=12)
ax.set_title('Correct Class Probability vs Threshold', fontsize=12)
ax.set_xscale('symlog', linthresh=1)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Fraction of channels zeroed vs threshold
ax = axes[2]
ax.plot([r['threshold'] for r in results], [r['frac_zeroed']*100 for r in results], 'r-o', linewidth=2)
# Mark where prediction changes
for i, r in enumerate(results):
    if not r['is_correct']:
        ax.axvline(x=r['threshold'], color='red', linestyle='--', alpha=0.3)
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('% Channels Zeroed', fontsize=12)
ax.set_title('Sparsity vs Threshold', fontsize=12)
ax.set_xscale('symlog', linthresh=1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cifar10_hoyer_ablation_study.png', dpi=150)
print("\nSaved: cifar10_hoyer_ablation_study.png")
plt.show()

# Summary
print(f"\nAblation Summary for sample {sample_idx} ({CIFAR_CLASSES[y_sample]}):")
print("-" * 60)
for r in results:
    status = "✓" if r['is_correct'] else "✗"
    print(f"  Thresh {r['threshold']:4d}: {r['frac_zeroed']*100:5.1f}% zeroed, "
          f"active={r['n_active']:3d}, pred={CIFAR_CLASSES[r['pred_class']]:10s} {status}")

# %%
# Find samples where bilinear blocks matter:
# - Full model gets correct answer
# - Embed @ head alone gets wrong answer

print("\n" + "="*60)
print("Finding samples where bilinear blocks matter...")
print("="*60)

# Get predictions with all channels zeroed (embed @ head only)
ablation_model.threshold = float('inf')  # Zero everything

embed_only_preds = []
with torch.no_grad():
    for i in range(len(test_data)):
        x_i, y_i = test_data[i]
        x_i = x_i.unsqueeze(0).to(DEVICE)
        logits, _, _, _ = ablation_model(x_i)
        embed_only_preds.append(logits.argmax(1).item())

embed_only_preds = torch.tensor(embed_only_preds)

# Find samples where full model correct, embed-only wrong
full_correct = (all_preds == all_labels)
embed_wrong = (embed_only_preds != all_labels)
bilinear_matters = full_correct & embed_wrong

bilinear_matters_indices = torch.where(bilinear_matters)[0]
print(f"Found {len(bilinear_matters_indices)} samples where bilinear blocks flip prediction to correct")

# %%
# Ablation study on 3 samples where bilinear blocks matter

n_examples = min(5, len(bilinear_matters_indices))
example_indices = bilinear_matters_indices[:n_examples].tolist()

all_example_results = []

for ex_i, sample_idx in enumerate(example_indices):
    x_sample, y_sample = test_data[sample_idx]
    x_sample = x_sample.unsqueeze(0).to(DEVICE)

    # Get embed-only logits to find top classes
    ablation_model.threshold = float('inf')
    with torch.no_grad():
        embed_logits, _, _, _ = ablation_model(x_sample)
    embed_logits = embed_logits[0]  # [10]

    # Get top 3 classes from embed-only, excluding correct class
    # Mask out correct class
    embed_logits_masked = embed_logits.clone()
    embed_logits_masked[y_sample] = float('-inf')
    top3_embed = torch.topk(embed_logits_masked, k=3).indices.tolist()

    embed_pred = top3_embed[0]  # Top-1 from embed (wrong)
    embed_2nd = top3_embed[1]   # 2nd from embed
    embed_3rd = top3_embed[2]   # 3rd from embed

    full_pred = all_preds[sample_idx].item()

    print(f"\n{'='*60}")
    print(f"Example {ex_i+1}: Sample {sample_idx}")
    print(f"  True class: {CIFAR_CLASSES[y_sample]}")
    print(f"  Embed-only top-1: {CIFAR_CLASSES[embed_pred]} (WRONG)")
    print(f"  Embed-only top-2: {CIFAR_CLASSES[embed_2nd]}")
    print(f"  Embed-only top-3: {CIFAR_CLASSES[embed_3rd]}")
    print(f"  Full model pred: {CIFAR_CLASSES[full_pred]} (CORRECT)")
    print("="*60)

    # Sweep thresholds
    thresholds = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    results = []

    for thresh in thresholds:
        ablation_model.threshold = thresh
        with torch.no_grad():
            logits, h_list, n_zeroed, n_total = ablation_model(x_sample)

        probs = F.softmax(logits, dim=1)[0]
        pred_class = logits.argmax(1).item()
        correct_logit = logits[0, y_sample].item()
        correct_prob = probs[y_sample].item()
        # Track embed-only top 3 classes
        embed_class_logit = logits[0, embed_pred].item()
        embed_class_prob = probs[embed_pred].item()
        embed_2nd_logit = logits[0, embed_2nd].item()
        embed_2nd_prob = probs[embed_2nd].item()
        embed_3rd_logit = logits[0, embed_3rd].item()
        embed_3rd_prob = probs[embed_3rd].item()
        frac_zeroed = n_zeroed / n_total

        results.append({
            'threshold': thresh,
            'correct_logit': correct_logit,
            'correct_prob': correct_prob,
            'embed_class_logit': embed_class_logit,
            'embed_class_prob': embed_class_prob,
            'embed_2nd_logit': embed_2nd_logit,
            'embed_2nd_prob': embed_2nd_prob,
            'embed_3rd_logit': embed_3rd_logit,
            'embed_3rd_prob': embed_3rd_prob,
            'pred_class': pred_class,
            'is_correct': pred_class == y_sample,
            'frac_zeroed': frac_zeroed,
            'n_active': n_total - n_zeroed,
        })

        status = "✓" if pred_class == y_sample else "✗"
        print(f"  Thresh={thresh:5d}: {frac_zeroed*100:5.1f}% zeroed, "
              f"active={n_total-n_zeroed:3d}, pred={CIFAR_CLASSES[pred_class]:10s} {status}")

    all_example_results.append({
        'sample_idx': sample_idx,
        'true_class': y_sample,
        'embed_pred': embed_pred,
        'embed_2nd': embed_2nd,
        'embed_3rd': embed_3rd,
        'results': results
    })

# Now plot with 4 columns - include top 3 embed classes
fig, axes = plt.subplots(n_examples, 4, figsize=(20, 5*n_examples))
if n_examples == 1:
    axes = axes.reshape(1, -1)

for ex_i, ex in enumerate(all_example_results):
    sample_idx = ex['sample_idx']
    y_sample = ex['true_class']
    embed_pred = ex['embed_pred']
    embed_2nd = ex['embed_2nd']
    embed_3rd = ex['embed_3rd']
    results = ex['results']

    # Plot 1: Logits for correct + top 3 embed classes
    ax = axes[ex_i, 0]
    ax.plot([r['threshold'] for r in results], [r['correct_logit'] for r in results],
            'b-o', linewidth=2, alpha=0.9, label=f'{CIFAR_CLASSES[y_sample]} (correct)')
    ax.plot([r['threshold'] for r in results], [r['embed_class_logit'] for r in results],
            'r-s', linewidth=2, alpha=0.9, label=f'{CIFAR_CLASSES[embed_pred]} (embed #1)')
    ax.plot([r['threshold'] for r in results], [r['embed_2nd_logit'] for r in results],
            'orange', marker='^', linewidth=1.5, alpha=0.7, label=f'{CIFAR_CLASSES[embed_2nd]} (embed #2)')
    ax.plot([r['threshold'] for r in results], [r['embed_3rd_logit'] for r in results],
            'purple', marker='d', linewidth=1.5, alpha=0.7, label=f'{CIFAR_CLASSES[embed_3rd]} (embed #3)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Logit')
    ax.set_title(f'Sample {sample_idx}: Logits')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Probabilities for all 4 classes
    ax = axes[ex_i, 1]
    ax.plot([r['threshold'] for r in results], [r['correct_prob'] for r in results],
            'b-o', linewidth=2, alpha=0.9, label=f'{CIFAR_CLASSES[y_sample]} (correct)')
    ax.plot([r['threshold'] for r in results], [r['embed_class_prob'] for r in results],
            'r-s', linewidth=2, alpha=0.9, label=f'{CIFAR_CLASSES[embed_pred]} (embed #1)')
    ax.plot([r['threshold'] for r in results], [r['embed_2nd_prob'] for r in results],
            'orange', marker='^', linewidth=1.5, alpha=0.7, label=f'{CIFAR_CLASSES[embed_2nd]} (embed #2)')
    ax.plot([r['threshold'] for r in results], [r['embed_3rd_prob'] for r in results],
            'purple', marker='d', linewidth=1.5, alpha=0.7, label=f'{CIFAR_CLASSES[embed_3rd]} (embed #3)')
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Probability')
    ax.set_title('Class Probabilities')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Sparsity with correct/wrong coloring
    ax = axes[ex_i, 2]
    colors = ['green' if r['is_correct'] else 'red' for r in results]
    ax.scatter([r['threshold'] for r in results], [r['frac_zeroed']*100 for r in results],
               c=colors, s=100, zorder=5)
    ax.plot([r['threshold'] for r in results], [r['frac_zeroed']*100 for r in results], 'k-', alpha=0.3)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('% Channels Zeroed')
    ax.set_title('Sparsity (green=correct, red=wrong)')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # Plot 4: Number of active channels with correct/wrong coloring
    ax = axes[ex_i, 3]
    colors = ['green' if r['is_correct'] else 'red' for r in results]
    ax.scatter([r['threshold'] for r in results], [r['n_active'] for r in results],
               c=colors, s=100, zorder=5)
    ax.plot([r['threshold'] for r in results], [r['n_active'] for r in results], 'k-', alpha=0.3)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('# Active Channels')
    ax.set_title('Active Channels (green=correct, red=wrong)')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cifar10_hoyer_ablation_bilinear_matters.png', dpi=150)
print("\nSaved: cifar10_hoyer_ablation_bilinear_matters.png")
plt.show()

# %%
# Plot ALL 10 class logits for each sample
fig, axes = plt.subplots(n_examples, 2, figsize=(16, 5*n_examples))
if n_examples == 1:
    axes = axes.reshape(1, -1)

# Get all 10 logits for each sample at each threshold
for ex_i, ex in enumerate(all_example_results):
    sample_idx = ex['sample_idx']
    y_sample = ex['true_class']
    embed_pred = ex['embed_pred']

    x_sample, _ = test_data[sample_idx]
    x_sample = x_sample.unsqueeze(0).to(DEVICE)

    thresholds = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    all_logits = []  # [n_thresholds, 10]
    all_probs = []

    for thresh in thresholds:
        ablation_model.threshold = thresh
        with torch.no_grad():
            logits, _, _, _ = ablation_model(x_sample)
        all_logits.append(logits[0].cpu().numpy())
        all_probs.append(F.softmax(logits, dim=1)[0].cpu().numpy())

    all_logits = np.array(all_logits)  # [n_thresholds, 10]
    all_probs = np.array(all_probs)

    # Plot logits for all 10 classes
    ax = axes[ex_i, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in range(10):
        alpha = 1.0 if c == y_sample else (0.8 if c == embed_pred else 0.4)
        linewidth = 2.5 if c == y_sample else (2 if c == embed_pred else 1)
        linestyle = '-' if c in [y_sample, embed_pred] else '--'
        ax.plot(thresholds, all_logits[:, c], color=colors[c],
                linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                label=f'{CIFAR_CLASSES[c]}' + (' ✓' if c == y_sample else (' ✗' if c == embed_pred else '')))
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Logit')
    ax.set_title(f'Sample {sample_idx}: All 10 Class Logits')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot probs for all 10 classes
    ax = axes[ex_i, 1]
    for c in range(10):
        alpha = 1.0 if c == y_sample else (0.8 if c == embed_pred else 0.4)
        linewidth = 2.5 if c == y_sample else (2 if c == embed_pred else 1)
        linestyle = '-' if c in [y_sample, embed_pred] else '--'
        ax.plot(thresholds, all_probs[:, c], color=colors[c],
                linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                label=f'{CIFAR_CLASSES[c]}' + (' ✓' if c == y_sample else (' ✗' if c == embed_pred else '')))
    ax.axhline(y=0.1, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Probability')
    ax.set_title(f'Sample {sample_idx}: All 10 Class Probabilities')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cifar10_hoyer_ablation_all10_classes.png', dpi=150)
print("Saved: cifar10_hoyer_ablation_all10_classes.png")
plt.show()

# %%
# For each sample where bilinear matters:
# 1. D @ head composition ordered by activation amount
# 2. D @ head scaled by activation amount

for ex_i, ex in enumerate(all_example_results):
    sample_idx = ex['sample_idx']
    y_sample = ex['true_class']
    embed_pred = ex['embed_pred']

    # Get activations for this sample
    x_sample, _ = test_data[sample_idx]
    x_sample = x_sample.unsqueeze(0).to(DEVICE)

    # Get embed-only baseline logits
    ablation_model.threshold = float('inf')
    with torch.no_grad():
        embed_baseline_logits, _, _, _ = ablation_model(x_sample)
    embed_baseline_correct = embed_baseline_logits[0, y_sample].item()
    embed_baseline_embed = embed_baseline_logits[0, embed_pred].item()

    with torch.no_grad():
        _, h_list = model(x_sample)

    # Concatenate activations from all blocks
    h_cat = torch.cat(h_list, dim=1)[0].cpu()  # [64]
    h_abs = h_cat.abs()

    # Get direct effects (D @ head) for each channel
    # direct_effects_flat is [total_channels, n_classes]

    # Sort channels by activation magnitude
    sorted_indices = torch.argsort(h_abs, descending=True)

    # Get effects for correct class and embed-only class
    correct_effects = direct_effects_flat[:, y_sample].numpy()
    embed_effects = direct_effects_flat[:, embed_pred].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: D @ head ordered by activation (unscaled)
    ax = axes[0]
    x_pos = np.arange(64)

    # Reorder by activation magnitude
    correct_ordered = correct_effects[sorted_indices]
    embed_ordered = embed_effects[sorted_indices]
    h_ordered = h_abs[sorted_indices].numpy()

    width = 0.35
    ax.bar(x_pos - width/2, correct_ordered, width, label=f'{CIFAR_CLASSES[y_sample]} (correct)', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, embed_ordered, width, label=f'{CIFAR_CLASSES[embed_pred]} (embed-only)', color='red', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Channel (sorted by |activation|)')
    ax.set_ylabel('D @ head (unscaled)')
    ax.set_title(f'Sample {sample_idx}: Direct Logit Effect (ordered by |h|)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add activation magnitude on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x_pos, h_ordered, 'k--', alpha=0.5, label='|activation|')
    ax2.set_ylabel('|activation|', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Plot 2: D @ head SCALED by activation
    ax = axes[1]

    # Scale by actual activation (not abs, to preserve sign)
    h_signed = h_cat[sorted_indices].numpy()
    correct_scaled = correct_ordered * h_signed
    embed_scaled = embed_ordered * h_signed

    ax.bar(x_pos - width/2, correct_scaled, width, label=f'{CIFAR_CLASSES[y_sample]} (correct)', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, embed_scaled, width, label=f'{CIFAR_CLASSES[embed_pred]} (embed-only)', color='red', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Channel (sorted by |activation|)')
    ax.set_ylabel('h * (D @ head)')
    ax.set_title(f'Sample {sample_idx}: Actual Logit Contribution (h × effect)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show cumulative contribution
    cum_correct = np.cumsum(correct_scaled)
    cum_embed = np.cumsum(embed_scaled)
    ax2 = ax.twinx()
    ax2.plot(x_pos, cum_correct, 'b-', alpha=0.6, linewidth=2, label='cumsum correct')
    ax2.plot(x_pos, cum_embed, 'r-', alpha=0.6, linewidth=2, label='cumsum embed')
    # Add dashed lines offset by embed baseline
    ax2.plot(x_pos, cum_correct + embed_baseline_correct, 'b--', alpha=0.3, linewidth=2, label='+ embed baseline')
    ax2.plot(x_pos, cum_embed + embed_baseline_embed, 'r--', alpha=0.3, linewidth=2)
    ax2.set_ylabel('Cumulative contribution', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    plt.savefig(f'cifar10_hoyer_sample{sample_idx}_head_composition.png', dpi=150)
    print(f"Saved: cifar10_hoyer_sample{sample_idx}_head_composition.png")
    plt.show()

    # Print summary
    total_correct = correct_scaled.sum()
    total_embed = embed_scaled.sum()
    print(f"  Sample {sample_idx} ({CIFAR_CLASSES[y_sample]} vs {CIFAR_CLASSES[embed_pred]}):")
    print(f"    Total contribution to correct class: {total_correct:.2f}")
    print(f"    Total contribution to embed class: {total_embed:.2f}")
    print(f"    Difference (correct - embed): {total_correct - total_embed:.2f}")

# Summary: find threshold where prediction flips
print("\n" + "="*60)
print("SUMMARY: Threshold where prediction flips from correct to wrong")
print("="*60)
for ex in all_example_results:
    sample_idx = ex['sample_idx']
    true_class = CIFAR_CLASSES[ex['true_class']]
    results = ex['results']

    # Find first threshold where prediction becomes wrong
    flip_thresh = None
    for r in results:
        if not r['is_correct']:
            flip_thresh = r['threshold']
            break

    if flip_thresh is not None:
        # Find how many channels were active just before flip
        for i, r in enumerate(results):
            if r['threshold'] == flip_thresh and i > 0:
                prev_active = results[i-1]['n_active']
                break
        print(f"  Sample {sample_idx} ({true_class}): flips at thresh={flip_thresh}, "
              f"needs >{results[i-1]['n_active']} active channels")
    else:
        print(f"  Sample {sample_idx} ({true_class}): never flips (robust)")

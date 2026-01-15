"""
CIFAR-10 BilinearMixerNet utilities and analysis functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']


# =============================================================================
# Model Definitions
# =============================================================================

class BilinearMixer(nn.Module):
    """Token mixing via bilinear operation."""
    def __init__(self, n_patches, hidden_dim, embed_dim):
        super().__init__()
        self.L = nn.Linear(n_patches, hidden_dim, bias=False)
        self.R = nn.Linear(n_patches, hidden_dim, bias=False)
        self.D = nn.Linear(hidden_dim, n_patches, bias=False)

    def forward(self, x):
        # x: [B, n_patches, embed_dim]
        y = x.transpose(1, 2)  # [B, embed_dim, n_patches]
        u = self.L(y)  # [B, embed_dim, hidden_dim]
        v = self.R(y)
        h = u * v  # element-wise product
        out = self.D(h)  # [B, embed_dim, n_patches]
        out = out.transpose(1, 2)  # [B, n_patches, embed_dim]
        return x + out, h  # Return hidden activations too


class BilinearLayer(nn.Module):
    """Channel mixing via bilinear operation."""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.L = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.R = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.D = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: [B, n_patches, embed_dim]
        u = self.L(x)  # [B, n_patches, hidden_dim]
        v = self.R(x)
        h = u * v
        out = self.D(h)
        return x + out, h


class BilinearMixerNet(nn.Module):
    def __init__(self, patch_size=4, embed_dim=128, mixer_hidden_dim=64,
                 layer_hidden_dim=64, n_blocks=1, n_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (32 // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        self.mixers = nn.ModuleList()
        self.layers = nn.ModuleList()
        for _ in range(n_blocks):
            self.mixers.append(BilinearMixer(self.n_patches, mixer_hidden_dim, embed_dim))
            self.layers.append(BilinearLayer(embed_dim, layer_hidden_dim))

        self.head = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim
        self.mixer_hidden_dim = mixer_hidden_dim
        self.layer_hidden_dim = layer_hidden_dim
        self.n_blocks = n_blocks

    def forward(self, x):
        B = x.shape[0]
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, self.n_patches, -1)

        x = self.patch_embed(x)

        all_hidden = []
        for mixer, layer in zip(self.mixers, self.layers):
            x, h_mixer = mixer(x)
            x, h_layer = layer(x)
            all_hidden.append(h_mixer)  # [B, embed_dim, mixer_hidden]
            all_hidden.append(h_layer)  # [B, n_patches, layer_hidden]

        x = x.mean(dim=1)
        return self.head(x), all_hidden


class BilinearMixerNetWithAblation(nn.Module):
    """BilinearMixerNet that can zero out channels below threshold."""
    def __init__(self, base_model, threshold=0.0):
        super().__init__()
        self.patch_size = base_model.patch_size
        self.n_patches = base_model.n_patches
        self.patch_embed = base_model.patch_embed
        self.mixers = base_model.mixers
        self.layers = base_model.layers
        self.head = base_model.head
        self.threshold = threshold

    def forward(self, x):
        B = x.shape[0]
        p = self.patch_size

        # Patchify
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, self.n_patches, -1)
        x = self.patch_embed(x)

        all_h = []
        n_zeroed = 0
        n_total = 0

        for mixer, layer in zip(self.mixers, self.layers):
            # Mixer forward with ablation
            y = x.transpose(1, 2)  # [B, embed_dim, n_patches]
            u_m = mixer.L(y)
            v_m = mixer.R(y)
            h_m = u_m * v_m

            # Zero out below threshold
            mask_m = h_m.abs() >= self.threshold
            h_m_masked = h_m * mask_m.float()
            n_zeroed += (~mask_m).sum().item()
            n_total += mask_m.numel()

            out_m = mixer.D(h_m_masked)
            x = x + out_m.transpose(1, 2)
            all_h.append(h_m_masked)

            # Layer forward with ablation
            u_l = layer.L(x)
            v_l = layer.R(x)
            h_l = u_l * v_l

            mask_l = h_l.abs() >= self.threshold
            h_l_masked = h_l * mask_l.float()
            n_zeroed += (~mask_l).sum().item()
            n_total += mask_l.numel()

            out_l = layer.D(h_l_masked)
            x = x + out_l
            all_h.append(h_l_masked)

        x = x.mean(dim=1)
        return self.head(x), all_h, n_zeroed, n_total


# =============================================================================
# Data Loading
# =============================================================================

def get_cifar10_loaders(batch_size=256, num_workers=4):
    """Load CIFAR-10 train and test dataloaders."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_data, test_data


def get_cifar10_test_loader(batch_size=256, num_workers=4):
    """Load CIFAR-10 test dataloader only."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader, test_data


def load_model(model_path, mixer_hidden=64, layer_hidden=64, embed_dim=128, n_blocks=1, device=None):
    """Load a trained BilinearMixerNet model."""
    if device is None:
        device = DEVICE

    model = BilinearMixerNet(
        patch_size=4,
        embed_dim=embed_dim,
        mixer_hidden_dim=mixer_hidden,
        layer_hidden_dim=layer_hidden,
        n_blocks=n_blocks,
        n_classes=10,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# =============================================================================
# Weight Sparsity Analysis
# =============================================================================

def analyze_weight_sparsity(model):
    """Analyze sparsity of L, R, D weight matrices."""
    results = {}

    for name, param in model.named_parameters():
        if 'weight' in name and any(x in name for x in ['L.', 'R.', 'D.']):
            total = param.numel()
            nonzero = (param.data != 0).sum().item()
            sparsity = 1 - nonzero / total

            results[name] = {
                'shape': tuple(param.shape),
                'total': total,
                'nonzero': nonzero,
                'sparsity': sparsity,
            }

    return results


def print_weight_sparsity(weight_sparsity):
    """Print weight sparsity analysis results."""
    print("\nWeight Sparsity Analysis:")
    print("=" * 70)
    total_params = 0
    total_nonzero = 0
    for name, info in weight_sparsity.items():
        print(f"  {name:40s} {str(info['shape']):15s} "
              f"sparsity={info['sparsity']:.1%} ({info['nonzero']}/{info['total']})")
        total_params += info['total']
        total_nonzero += info['nonzero']

    overall_sparsity = 1 - total_nonzero / total_params
    print(f"\n  Overall L/R/D sparsity: {overall_sparsity:.1%} ({total_nonzero}/{total_params})")
    return overall_sparsity


def plot_weight_matrices(model, save_path='cifar10_images/hoyer_imp_weight_sparsity.png'):
    """Visualize the sparsity pattern of weight matrices."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    layer_names = [
        ('mixers.0.L', model.mixers[0].L.weight),
        ('mixers.0.R', model.mixers[0].R.weight),
        ('mixers.0.D', model.mixers[0].D.weight),
        ('layers.0.L', model.layers[0].L.weight),
        ('layers.0.R', model.layers[0].R.weight),
        ('layers.0.D', model.layers[0].D.weight),
    ]

    for ax, (name, W) in zip(axes.flat, layer_names):
        W_np = W.detach().cpu().numpy()

        # Show non-zero pattern
        im = ax.imshow(W_np != 0, aspect='auto', cmap='Blues')
        ax.set_title(f'{name}\n{W_np.shape}, sparsity={(W_np==0).mean():.1%}')
        ax.set_xlabel('Input dim')
        ax.set_ylabel('Output dim')

    plt.suptitle('Weight Sparsity Patterns (blue = non-zero)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# Activation Collection
# =============================================================================

@torch.no_grad()
def collect_activations(model, loader, device=None):
    """Collect all channel activations and predictions."""
    if device is None:
        device = DEVICE

    all_mixer_h = []
    all_layer_h = []
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, h_list = model(x)

        h_mixer = h_list[0]  # [B, 128, 64]
        h_layer = h_list[1]  # [B, 64, 64]

        all_mixer_h.append(h_mixer.cpu())
        all_layer_h.append(h_layer.cpu())
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())

    all_mixer_h = torch.cat(all_mixer_h, dim=0)
    all_layer_h = torch.cat(all_layer_h, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_mixer_h, all_layer_h, all_preds, all_labels


# =============================================================================
# Hoyer Sparsity Statistics
# =============================================================================

def compute_hoyer_stats(all_mixer_h, all_layer_h):
    """Compute Hoyer sparsity statistics."""
    N = all_mixer_h.shape[0]
    mixer_flat = all_mixer_h.reshape(N, -1)
    layer_flat = all_layer_h.reshape(N, -1)
    all_flat = torch.cat([mixer_flat, layer_flat], dim=1)

    n = all_flat.shape[1]
    l1 = all_flat.abs().sum(dim=1)
    l2 = torch.sqrt((all_flat ** 2).sum(dim=1) + 1e-8)
    sqrt_n = np.sqrt(n)
    hoyer = (sqrt_n - l1 / (l2 + 1e-8)) / (sqrt_n - 1 + 1e-8)

    sqrt_k = sqrt_n * (1 - hoyer) + hoyer
    effective_k = sqrt_k ** 2

    return {
        'hoyer_mean': hoyer.mean().item(),
        'hoyer_std': hoyer.std().item(),
        'hoyer_min': hoyer.min().item(),
        'hoyer_max': hoyer.max().item(),
        'effective_k_mean': effective_k.mean().item(),
        'effective_k_std': effective_k.std().item(),
        'n_total': n,
        'hoyer_per_sample': hoyer,
        'effective_k_per_sample': effective_k,
    }


def print_hoyer_stats(hoyer_stats):
    """Print Hoyer sparsity statistics."""
    print("\nHoyer Sparsity Statistics:")
    print("=" * 60)
    print(f"  Total hidden dims: {hoyer_stats['n_total']}")
    print(f"  Hoyer mean: {hoyer_stats['hoyer_mean']:.4f}")
    print(f"  Hoyer std:  {hoyer_stats['hoyer_std']:.4f}")
    print(f"  Hoyer range: [{hoyer_stats['hoyer_min']:.4f}, {hoyer_stats['hoyer_max']:.4f}]")
    print(f"  Effective active channels: {hoyer_stats['effective_k_mean']:.1f} ± {hoyer_stats['effective_k_std']:.1f}")
    print(f"  % of total: {100 * hoyer_stats['effective_k_mean'] / hoyer_stats['n_total']:.2f}%")


def plot_hoyer_distribution(hoyer_stats, save_path='cifar10_images/hoyer_imp_sparsity_dist.png'):
    """Plot Hoyer sparsity and effective channels distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hoyer histogram
    ax = axes[0]
    ax.hist(hoyer_stats['hoyer_per_sample'].numpy(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=hoyer_stats['hoyer_mean'], color='red', linestyle='--',
               label=f"Mean={hoyer_stats['hoyer_mean']:.3f}")
    ax.set_xlabel('Hoyer Sparsity')
    ax.set_ylabel('Count')
    ax.set_title('Hoyer Sparsity Distribution Across Test Set')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effective channels histogram
    ax = axes[1]
    ax.hist(hoyer_stats['effective_k_per_sample'].numpy(), bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=hoyer_stats['effective_k_mean'], color='red', linestyle='--',
               label=f"Mean={hoyer_stats['effective_k_mean']:.1f}")
    ax.set_xlabel('Effective Active Channels')
    ax.set_ylabel('Count')
    ax.set_title(f'Effective Active Channels (out of {hoyer_stats["n_total"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# Channel Activation Frequency
# =============================================================================

def plot_channel_activation_freq(all_mixer_h, all_layer_h, threshold=0.1,
                                  save_path='cifar10_images/hoyer_imp_activation_freq.png'):
    """Plot activation frequency for mixer and layer channels."""
    # Mixer: [N, 128, 64] -> frequency per (embed_pos, channel)
    mixer_active = (all_mixer_h.abs() > threshold).float()
    mixer_freq = mixer_active.mean(dim=0).numpy()

    # Layer: [N, 64, 64] -> frequency per (patch, channel)
    layer_active = (all_layer_h.abs() > threshold).float()
    layer_freq = layer_active.mean(dim=0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(mixer_freq, aspect='auto', cmap='hot')
    ax.set_xlabel('Hidden Channel')
    ax.set_ylabel('Embed Position')
    ax.set_title(f'Mixer Activation Frequency (thresh={threshold})')
    plt.colorbar(im, ax=ax, label='Frequency')

    ax = axes[1]
    im = ax.imshow(layer_freq, aspect='auto', cmap='hot')
    ax.set_xlabel('Hidden Channel')
    ax.set_ylabel('Patch Position')
    ax.set_title(f'Layer Activation Frequency (thresh={threshold})')
    plt.colorbar(im, ax=ax, label='Frequency')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

    # Print stats
    print(f"\nActivation frequency stats (threshold={threshold}):")
    print(f"  Mixer: {(mixer_freq > 0).sum()} / {mixer_freq.size} positions ever active")
    print(f"  Mixer mean freq: {mixer_freq.mean():.4f}")
    print(f"  Layer: {(layer_freq > 0).sum()} / {layer_freq.size} positions ever active")
    print(f"  Layer mean freq: {layer_freq.mean():.4f}")

    return mixer_freq, layer_freq


# =============================================================================
# Direct Logit Effects
# =============================================================================

def compute_direct_logit_effects(model):
    """Compute direct effect of layer hidden channels on output classes."""
    head_W = model.head.weight.detach()  # [10, embed_dim]
    layer_D = model.layers[0].D.weight.detach()  # [embed_dim, layer_hidden]
    layer_effects = head_W @ layer_D  # [10, layer_hidden]
    return layer_effects.cpu()


def plot_direct_logit_effects(layer_direct_effects, save_path='cifar10_images/hoyer_imp_direct_effects.png'):
    """Plot direct logit effects heatmap."""
    print(f"\nLayer direct effects shape: {layer_direct_effects.shape}")

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(layer_direct_effects.numpy(), aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Hidden Channel')
    ax.set_ylabel('Class')
    ax.set_yticks(range(10))
    ax.set_yticklabels(CIFAR_CLASSES)
    ax.set_title('Layer Channel Direct Effect on Class Logits (head @ D)')
    plt.colorbar(im, ax=ax, label='Effect')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

    # Print top channels
    effect_magnitude = layer_direct_effects.abs().max(dim=0).values
    top_channels = torch.argsort(effect_magnitude, descending=True)[:10]
    print("\nTop 10 channels by max effect magnitude:")
    for i, ch in enumerate(top_channels):
        max_class = layer_direct_effects[:, ch].abs().argmax().item()
        max_effect = layer_direct_effects[max_class, ch].item()
        print(f"  {i+1}. Channel {ch.item()}: max effect {max_effect:.3f} on {CIFAR_CLASSES[max_class]}")


# =============================================================================
# Sample Analysis
# =============================================================================

def analyze_sample(model, x, y, layer_direct_effects, device=None):
    """Analyze a single sample's activations and contributions."""
    if device is None:
        device = DEVICE

    model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)
        logits, h_list = model(x)
        pred = logits.argmax(1).item()

    h_mixer = h_list[0].squeeze(0).cpu()
    h_layer = h_list[1].squeeze(0).cpu()
    logits = logits.squeeze(0).cpu()

    h_layer_mean = h_layer.mean(dim=0)
    layer_contrib = h_layer_mean.unsqueeze(1) * layer_direct_effects.T

    return {
        'h_mixer': h_mixer.numpy(),
        'h_layer': h_layer.numpy(),
        'h_layer_mean': h_layer_mean.numpy(),
        'logits': logits.numpy(),
        'pred': pred,
        'true': y,
        'layer_contrib': layer_contrib.numpy(),
    }


def plot_sample_analysis(model, test_data, correct_indices, layer_direct_effects, device=None,
                         n_samples=3, save_path='cifar10_images/hoyer_imp_sample_analysis.png'):
    """Plot analysis for correctly classified samples."""
    if device is None:
        device = DEVICE

    print(f"\nCorrectly classified: {len(correct_indices)} / {len(test_data)}")

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))

    for i in range(n_samples):
        sample_idx = correct_indices[i].item()
        x_sample, y_sample = test_data[sample_idx]

        analysis = analyze_sample(model, x_sample, y_sample, layer_direct_effects, device)

        # Plot 1: Layer hidden activations (mean over patches)
        ax = axes[i, 0]
        ax.bar(range(64), analysis['h_layer_mean'])
        ax.set_xlabel('Channel')
        ax.set_ylabel('Mean Activation')
        ax.set_title(f'Sample {sample_idx}: Layer Activations\n(True: {CIFAR_CLASSES[y_sample]})')
        ax.grid(True, alpha=0.3)

        # Plot 2: Contribution to correct class
        ax = axes[i, 1]
        contrib_correct = analysis['layer_contrib'][:, y_sample]
        colors = ['blue' if c >= 0 else 'red' for c in contrib_correct]
        ax.bar(range(64), contrib_correct, color=colors, alpha=0.7)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Contribution')
        ax.set_title(f'Contribution to {CIFAR_CLASSES[y_sample]}')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 3: Logits
        ax = axes[i, 2]
        colors = ['green' if j == y_sample else 'gray' for j in range(10)]
        ax.barh(range(10), analysis['logits'], color=colors)
        ax.set_yticks(range(10))
        ax.set_yticklabels(CIFAR_CLASSES)
        ax.set_xlabel('Logit')
        ax.set_title(f'Predicted: {CIFAR_CLASSES[analysis["pred"]]}')
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# Per-Class Analysis
# =============================================================================

def per_class_analysis(all_preds, all_labels, hoyer_per_sample):
    """Analyze accuracy and sparsity per class."""
    results = {}

    for c in range(10):
        mask = (all_labels == c)
        n_samples = mask.sum().item()
        correct = ((all_preds == all_labels) & mask).sum().item()
        acc = correct / n_samples if n_samples > 0 else 0

        hoyer_class = hoyer_per_sample[mask]

        results[c] = {
            'n_samples': n_samples,
            'accuracy': acc,
            'hoyer_mean': hoyer_class.mean().item(),
            'hoyer_std': hoyer_class.std().item(),
        }

    return results


def print_per_class(per_class):
    """Print per-class analysis results."""
    print("\nPer-Class Analysis:")
    print("=" * 70)
    print(f"{'Class':<12} {'N':>6} {'Accuracy':>10} {'Hoyer Mean':>12} {'Hoyer Std':>12}")
    print("-" * 70)
    for c in range(10):
        r = per_class[c]
        print(f"{CIFAR_CLASSES[c]:<12} {r['n_samples']:>6} {r['accuracy']:>10.2%} "
              f"{r['hoyer_mean']:>12.4f} {r['hoyer_std']:>12.4f}")


def plot_per_class(per_class, save_path='cifar10_images/hoyer_imp_per_class.png'):
    """Plot per-class accuracy and Hoyer sparsity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy per class
    ax = axes[0]
    accs = [per_class[c]['accuracy'] for c in range(10)]
    ax.bar(range(10), accs, color='steelblue', alpha=0.7)
    ax.axhline(y=np.mean(accs), color='red', linestyle='--', label=f'Mean={np.mean(accs):.2%}')
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hoyer per class
    ax = axes[1]
    hoyers = [per_class[c]['hoyer_mean'] for c in range(10)]
    hoyer_stds = [per_class[c]['hoyer_std'] for c in range(10)]
    ax.bar(range(10), hoyers, yerr=hoyer_stds, color='orange', alpha=0.7, capsize=3)
    ax.axhline(y=np.mean(hoyers), color='red', linestyle='--', label=f'Mean={np.mean(hoyers):.3f}')
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha='right')
    ax.set_ylabel('Hoyer Sparsity')
    ax.set_title('Per-Class Hoyer Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# D-Vector Norms
# =============================================================================

def compute_d_vector_norms(model):
    """Compute L2 norms of D-vector columns."""
    results = {}

    mixer_D = model.mixers[0].D.weight.detach()
    mixer_norms = mixer_D.norm(dim=0).cpu().numpy()
    results['mixer'] = mixer_norms

    layer_D = model.layers[0].D.weight.detach()
    layer_norms = layer_D.norm(dim=0).cpu().numpy()
    results['layer'] = layer_norms

    return results


def plot_d_norms(d_norms, save_path='cifar10_images/hoyer_imp_d_norms.png'):
    """Plot D-vector norms for mixer and layer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(range(64), d_norms['mixer'], alpha=0.7)
    ax.set_xlabel('Hidden Channel')
    ax.set_ylabel('D-vector L2 Norm')
    ax.set_title(f"Mixer D-vector Norms (mean={d_norms['mixer'].mean():.3f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(range(64), d_norms['layer'], alpha=0.7, color='orange')
    ax.set_xlabel('Hidden Channel')
    ax.set_ylabel('D-vector L2 Norm')
    ax.set_title(f"Layer D-vector Norms (mean={d_norms['layer'].mean():.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

    print("\nD-vector norm stats:")
    print(f"  Mixer: min={d_norms['mixer'].min():.4f}, max={d_norms['mixer'].max():.4f}, "
          f"mean={d_norms['mixer'].mean():.4f}")
    print(f"  Layer: min={d_norms['layer'].min():.4f}, max={d_norms['layer'].max():.4f}, "
          f"mean={d_norms['layer'].mean():.4f}")


# =============================================================================
# Head Cosine Similarity
# =============================================================================

def plot_head_cosine_similarity(model, save_path='cifar10_images/hoyer_imp_head_cossim.png'):
    """Plot cosine similarity matrix of head output vectors."""
    head_W = model.head.weight.detach().cpu()
    head_W_norm = head_W / head_W.norm(dim=1, keepdim=True)
    cos_sim = head_W_norm @ head_W_norm.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos_sim.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR_CLASSES)
    ax.set_title('Cosine Similarity of Head Output Vectors', fontsize=14)

    for i in range(10):
        for j in range(10):
            val = cos_sim[i, j].item()
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

    off_diag_mask = ~torch.eye(10, dtype=bool)
    off_diag = cos_sim[off_diag_mask]
    print(f"\nHead cosine similarity stats (off-diagonal):")
    print(f"  Mean: {off_diag.mean():.4f}, Std: {off_diag.std():.4f}")
    print(f"  Min: {off_diag.min():.4f}, Max: {off_diag.max():.4f}")


# =============================================================================
# Ablation Studies
# =============================================================================

def run_single_ablation_study(model, test_data, sample_idx, device=None,
                               thresholds=None, save_path='cifar10_images/hoyer_imp_ablation_single.png'):
    """Run ablation study on a single sample."""
    if device is None:
        device = DEVICE
    if thresholds is None:
        thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

    ablation_model = BilinearMixerNetWithAblation(model, threshold=0.0)
    ablation_model.eval()

    x_sample, y_sample = test_data[sample_idx]
    x_sample_t = x_sample.unsqueeze(0).to(device)

    print(f"\nAblation study for sample {sample_idx}: True class = {CIFAR_CLASSES[y_sample]}")

    results = []
    for thresh in thresholds:
        ablation_model.threshold = thresh
        with torch.no_grad():
            logits, h_list, n_zeroed, n_total = ablation_model(x_sample_t)

        probs = F.softmax(logits, dim=1)[0]
        pred_class = logits.argmax(1).item()
        correct_prob = probs[y_sample].item()
        frac_zeroed = n_zeroed / n_total

        results.append({
            'threshold': thresh,
            'correct_logit': logits[0, y_sample].item(),
            'correct_prob': correct_prob,
            'pred_class': pred_class,
            'is_correct': pred_class == y_sample,
            'frac_zeroed': frac_zeroed,
            'n_active': n_total - n_zeroed,
        })

        status = "✓" if pred_class == y_sample else "✗"
        print(f"  Thresh={thresh:6.2f}: {frac_zeroed*100:5.1f}% zeroed, "
              f"pred={CIFAR_CLASSES[pred_class]:10s}, prob={correct_prob:.3f} {status}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot([r['threshold'] for r in results], [r['correct_logit'] for r in results], 'b-o', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Logit for Correct Class')
    ax.set_title(f'Correct Class ({CIFAR_CLASSES[y_sample]}) Logit')
    ax.set_xscale('symlog', linthresh=0.1)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot([r['threshold'] for r in results], [r['correct_prob'] for r in results], 'g-o', linewidth=2)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Probability')
    ax.set_title('Correct Class Probability')
    ax.set_xscale('symlog', linthresh=0.1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    colors = ['green' if r['is_correct'] else 'red' for r in results]
    ax.scatter([r['threshold'] for r in results], [r['frac_zeroed']*100 for r in results],
               c=colors, s=100, zorder=5)
    ax.plot([r['threshold'] for r in results], [r['frac_zeroed']*100 for r in results], 'k-', alpha=0.3)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('% Channels Zeroed')
    ax.set_title('Sparsity (green=correct)')
    ax.set_xscale('symlog', linthresh=0.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

    return results


def find_bilinear_matters_samples(model, test_data, all_preds, all_labels, device=None):
    """Find samples where bilinear blocks flip prediction to correct."""
    if device is None:
        device = DEVICE

    print("\n" + "=" * 60)
    print("Finding samples where bilinear blocks matter...")
    print("=" * 60)

    ablation_model = BilinearMixerNetWithAblation(model, threshold=float('inf'))
    ablation_model.eval()

    embed_only_preds = []
    with torch.no_grad():
        for i in range(len(test_data)):
            x_i, _ = test_data[i]
            x_i = x_i.unsqueeze(0).to(device)
            logits, _, _, _ = ablation_model(x_i)
            embed_only_preds.append(logits.argmax(1).item())

    embed_only_preds = torch.tensor(embed_only_preds)

    full_correct = (all_preds == all_labels)
    embed_wrong = (embed_only_preds != all_labels)
    bilinear_matters = full_correct & embed_wrong

    bilinear_matters_indices = torch.where(bilinear_matters)[0]
    print(f"Found {len(bilinear_matters_indices)} samples where bilinear blocks flip prediction to correct")

    embed_only_acc = (embed_only_preds == all_labels).float().mean()
    full_acc = (all_preds == all_labels).float().mean()
    print(f"Embed-only accuracy: {embed_only_acc:.2%}")
    print(f"Full model accuracy: {full_acc:.2%}")
    print(f"Improvement from bilinear blocks: {(full_acc - embed_only_acc)*100:.1f}pp")

    return bilinear_matters_indices, embed_only_preds, embed_only_acc


def run_detailed_ablation(model, test_data, bilinear_matters_indices, all_preds, device=None,
                          n_examples=3, thresholds=None,
                          save_path='cifar10_images/hoyer_imp_ablation_bilinear_matters.png'):
    """Run detailed ablation on samples where bilinear blocks matter."""
    if device is None:
        device = DEVICE
    if thresholds is None:
        thresholds = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

    n_examples = min(n_examples, len(bilinear_matters_indices))
    if n_examples == 0:
        print("No samples where bilinear blocks matter found.")
        return []

    ablation_model = BilinearMixerNetWithAblation(model, threshold=0.0)
    ablation_model.eval()

    example_indices = bilinear_matters_indices[:n_examples].tolist()
    all_example_results = []

    for ex_i, sample_idx in enumerate(example_indices):
        x_sample, y_sample = test_data[sample_idx]
        x_sample_t = x_sample.unsqueeze(0).to(device)

        # Get embed-only prediction
        ablation_model.threshold = float('inf')
        with torch.no_grad():
            embed_logits, _, _, _ = ablation_model(x_sample_t)
        embed_pred = embed_logits.argmax(1).item()

        full_pred = all_preds[sample_idx].item()

        print(f"\n{'=' * 60}")
        print(f"Example {ex_i + 1}: Sample {sample_idx}")
        print(f"  True class: {CIFAR_CLASSES[y_sample]}")
        print(f"  Embed-only pred: {CIFAR_CLASSES[embed_pred]} (WRONG)")
        print(f"  Full model pred: {CIFAR_CLASSES[full_pred]} (CORRECT)")
        print("=" * 60)

        results = []
        for thresh in thresholds:
            ablation_model.threshold = thresh
            with torch.no_grad():
                logits, _, n_zeroed, n_total = ablation_model(x_sample_t)

            probs = F.softmax(logits, dim=1)[0]
            pred_class = logits.argmax(1).item()
            frac_zeroed = n_zeroed / n_total

            results.append({
                'threshold': thresh,
                'correct_logit': logits[0, y_sample].item(),
                'correct_prob': probs[y_sample].item(),
                'embed_logit': logits[0, embed_pred].item(),
                'embed_prob': probs[embed_pred].item(),
                'pred_class': pred_class,
                'is_correct': pred_class == y_sample,
                'frac_zeroed': frac_zeroed,
                'n_active': n_total - n_zeroed,
            })

            status = "✓" if pred_class == y_sample else "✗"
            print(f"  Thresh={thresh:6.2f}: {frac_zeroed*100:5.1f}% zeroed, "
                  f"pred={CIFAR_CLASSES[pred_class]:10s} {status}")

        all_example_results.append({
            'sample_idx': sample_idx,
            'true_class': y_sample,
            'embed_pred': embed_pred,
            'results': results
        })

    # Plot
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for ex_i, ex in enumerate(all_example_results):
        y_sample = ex['true_class']
        embed_pred = ex['embed_pred']
        results = ex['results']

        ax = axes[ex_i, 0]
        ax.plot([r['threshold'] for r in results], [r['correct_logit'] for r in results],
                'b-o', linewidth=2, label=f'{CIFAR_CLASSES[y_sample]} (correct)')
        ax.plot([r['threshold'] for r in results], [r['embed_logit'] for r in results],
                'r-s', linewidth=2, label=f'{CIFAR_CLASSES[embed_pred]} (embed)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Logit')
        ax.set_title(f'Sample {ex["sample_idx"]}: Logits')
        ax.set_xscale('symlog', linthresh=0.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[ex_i, 1]
        ax.plot([r['threshold'] for r in results], [r['correct_prob'] for r in results],
                'b-o', linewidth=2, label=f'{CIFAR_CLASSES[y_sample]} (correct)')
        ax.plot([r['threshold'] for r in results], [r['embed_prob'] for r in results],
                'r-s', linewidth=2, label=f'{CIFAR_CLASSES[embed_pred]} (embed)')
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')
        ax.set_xscale('symlog', linthresh=0.1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[ex_i, 2]
        colors = ['green' if r['is_correct'] else 'red' for r in results]
        ax.scatter([r['threshold'] for r in results], [r['n_active'] for r in results],
                   c=colors, s=100, zorder=5)
        ax.plot([r['threshold'] for r in results], [r['n_active'] for r in results], 'k-', alpha=0.3)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('# Active Channels')
        ax.set_title('Active Channels (green=correct)')
        ax.set_xscale('symlog', linthresh=0.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Threshold where prediction flips")
    print("=" * 60)
    for ex in all_example_results:
        sample_idx = ex['sample_idx']
        true_class = CIFAR_CLASSES[ex['true_class']]
        results = ex['results']

        flip_thresh = None
        for r in results:
            if not r['is_correct']:
                flip_thresh = r['threshold']
                break

        if flip_thresh is not None:
            print(f"  Sample {sample_idx} ({true_class}): flips at thresh={flip_thresh}")
        else:
            print(f"  Sample {sample_idx} ({true_class}): never flips (robust)")

    return all_example_results


# =============================================================================
# Final Summary
# =============================================================================

def print_final_summary(model_name, all_preds, all_labels, overall_sparsity, hoyer_stats, embed_only_acc):
    """Print final summary of analysis."""
    test_acc = (all_preds == all_labels).float().mean()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Weight Sparsity: {overall_sparsity:.1%}")
    print(f"Activation Hoyer: {hoyer_stats['hoyer_mean']:.3f}")
    print(f"Effective Active Channels: {hoyer_stats['effective_k_mean']:.1f} / {hoyer_stats['n_total']} "
          f"({100*hoyer_stats['effective_k_mean']/hoyer_stats['n_total']:.2f}%)")
    print(f"Embed-only Accuracy: {embed_only_acc:.2%}")
    print(f"Bilinear Block Contribution: +{(test_acc - embed_only_acc)*100:.1f}pp")

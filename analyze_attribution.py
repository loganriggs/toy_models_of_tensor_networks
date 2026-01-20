# %%
"""
Attribution Analysis Script

Interactive analysis of frozen-RMS trained model.
Run cells individually to explore attributions and ablations.
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

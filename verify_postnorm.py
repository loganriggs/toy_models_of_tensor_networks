"""
Verify post-norm RMSNorm attribution via ablation.

Post-norm: bilinear first, then normalize the residual.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

from cifar10_patch_rmsnorm import BilinearPatchNetRMSNorm


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = BilinearPatchNetRMSNorm(
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


def get_gradient_attribution(model, x, y, device):
    """Get gradient-based attribution with FROZEN RMS (not flowing gradients through RMS)."""
    model.eval()
    x_in = x.unsqueeze(0).to(device)

    # First forward pass - collect RMS values
    with torch.no_grad():
        residual = model.patch_embed(x_in).squeeze(0)
        rms_values = []

        for block in model.blocks:
            mlp_out, h = block.mlp(residual)
            residual = residual + mlp_out
            # Store RMS before normalizing
            rms = (residual.pow(2).mean(dim=-1, keepdim=True)).sqrt() + block.norm.eps
            rms_values.append(rms.clone())
            residual = block.norm(residual)

    # Second forward pass with frozen RMS for attribution
    residual = model.patch_embed(x_in).squeeze(0)

    all_h = []
    for li, block in enumerate(model.blocks):
        mlp_out, h = block.mlp(residual)
        h = h.detach().requires_grad_(True)
        all_h.append(h)

        mlp_out = block.mlp.D(h)
        residual = residual + mlp_out

        # Frozen RMS: treat rms as constant (no gradient flow through RMS calc)
        frozen_rms = rms_values[li].detach()
        gamma = block.norm.gamma
        residual = residual * (gamma / frozen_rms)

    x_pooled = residual.mean(dim=0)
    logit = model.head.weight[y] @ x_pooled
    if model.head.bias is not None:
        logit = logit + model.head.bias[y]

    # Backward through frozen network
    logit.backward()

    # Collect attributions
    all_attr = []
    for h in all_h:
        attr = (h.grad * h.detach())  # [n_patches, hidden_dim]
        all_attr.append(attr)

    return logit.item(), all_attr, [h.detach() for h in all_h]


def ablate_channel(model, x, y, layer_idx, patch_idx, channel_idx, device):
    """Compute logit with ONE channel zeroed out."""
    model.eval()
    x_in = x.unsqueeze(0).to(device)

    with torch.no_grad():
        residual = model.patch_embed(x_in).squeeze(0)

        for li, block in enumerate(model.blocks):
            # Post-norm: bilinear first
            mlp_out, h = block.mlp(residual)

            if li == layer_idx:
                h[patch_idx, channel_idx] = 0.0
                mlp_out = block.mlp.D(h)

            residual = residual + mlp_out
            residual = block.norm(residual)

        x_pooled = residual.mean(dim=0)
        logit = (model.head.weight[y] @ x_pooled).item()
        if model.head.bias is not None:
            logit += model.head.bias[y].item()

    return logit


def verify_sample(model, x, y, device, top_k=20):
    """Verify attribution for top-K channels."""
    full_logit, all_attr, all_h = get_gradient_attribution(model, x, y, device)

    n_patches = 64
    hidden_dim = model.hidden_dim
    n_per_layer = n_patches * hidden_dim

    # Flatten attributions
    contribs = torch.cat([attr.reshape(-1) for attr in all_attr])

    # Get top-K by magnitude
    sorted_idx = contribs.abs().argsort(descending=True)[:top_k]

    results = []

    for flat_idx in sorted_idx:
        flat_idx = flat_idx.item()

        layer_idx = flat_idx // n_per_layer
        remainder = flat_idx % n_per_layer
        patch_idx = remainder // hidden_dim
        channel_idx = remainder % hidden_dim

        attr = contribs[flat_idx].item()

        # Ablation test
        ablated_logit = ablate_channel(model, x, y, layer_idx, patch_idx, channel_idx, device)
        ablation_effect = full_logit - ablated_logit

        results.append({
            'layer': layer_idx,
            'patch': patch_idx,
            'channel': channel_idx,
            'attr': attr,
            'ablation_effect': ablation_effect,
            'ablation_error': abs(attr - ablation_effect),
            'rel_error': abs(attr - ablation_effect) / (abs(attr) + 1e-10),
        })

    return results, {
        'full_logit': full_logit,
        'total_pos': contribs[contribs > 0].sum().item(),
        'total_neg': contribs[contribs < 0].sum().item(),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    path = 'cifar10_models/patch_bilinear_rmsnorm_attr_10.0.pt'
    try:
        model, config = load_model(path)
        model = model.to(device)
    except FileNotFoundError:
        print(f"Model not found: {path}")
        return

    print(f"\n{'='*70}")
    print(f"Post-norm RMSNorm Attribution Verification (λ=10)")
    print('='*70)

    n_samples = 10
    all_layer0_errors = []
    all_layer1_errors = []

    for i in range(n_samples):
        x, y = test_dataset[i]
        results, info = verify_sample(model, x, y, device, top_k=20)

        if i == 0:
            print(f"\nSample {i}: full_logit={info['full_logit']:.3f}")
            print(f"Total pos/neg attr: {info['total_pos']:.3f} / {info['total_neg']:.3f}")
            print(f"\nTop 20 channels:")
            print(f"{'L':<3} {'P':<3} {'Ch':<4} {'Attr':>10} {'Ablation':>10} {'Error':>10} {'RelErr':>8}")
            print("-" * 60)

            for r in results:
                print(f"{r['layer']:<3} {r['patch']:<3} {r['channel']:<4} "
                      f"{r['attr']:>10.4f} {r['ablation_effect']:>10.4f} "
                      f"{r['ablation_error']:>10.6f} {r['rel_error']*100:>7.2f}%")

        for r in results:
            if r['layer'] == 0:
                all_layer0_errors.append(r['ablation_error'])
            else:
                all_layer1_errors.append(r['ablation_error'])

    print(f"\n--- Aggregated over {n_samples} samples ---")
    print(f"Layer 0 mean ablation error: {np.mean(all_layer0_errors):.6f}")
    print(f"Layer 1 mean ablation error: {np.mean(all_layer1_errors):.6f}")

    print(f"\n(Compare: pre-norm λ=10 had Layer0=3.6, Layer1=90.6)")
    print(f"(Compare: pre-norm λ=0 had Layer0=0.14, Layer1=0.02)")


if __name__ == '__main__':
    main()

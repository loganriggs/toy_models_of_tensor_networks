"""
Verify RMSNorm attribution accuracy via ablation.

Key insight: With frozen RMS, the normalization becomes just element-wise
scaling, which is linear. This should give much better causal accuracy
than LayerNorm or no-norm (which has bilinear second-order effects).
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


def compute_attribution_frozen_rms(model, x, y, device):
    """
    Compute attribution with frozen RMS values.

    The key is that RMSNorm with frozen RMS is just: y = x * (gamma / rms)
    This is element-wise scaling, so attribution passes through linearly.
    """
    model.eval()
    x_in = x.unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass storing RMS values
        logits, all_h, all_rms, final_rms = model.forward_with_rms(x_in)
        full_logit = logits[0, y].item()

        # Get class weights
        W_y = model.head.weight[y]  # [embed_dim]
        bias_y = model.head.bias[y].item() if model.head.bias is not None else 0.0

        # Work backwards to compute contributions

        # After final_norm and mean pool: logit = W @ mean(x_normed) + bias
        # x_normed = x * (gamma / final_rms)
        # So: logit = W @ mean(x * gamma / final_rms) + bias

        # Start from patch_embed
        patch_embed = model.patch_embed(x_in).squeeze(0)  # [n_patches, embed_dim]
        n_patches = patch_embed.shape[0]

        # Trace through layers, computing contributions
        residual = patch_embed.clone()
        all_contribs = []

        for layer_idx, block in enumerate(model.blocks):
            rms = all_rms[layer_idx]  # [n_patches, 1]

            # RMSNorm: x_normed = x * (gamma / rms)
            gamma = block.norm.gamma
            scale = gamma / rms  # [n_patches, embed_dim]

            # Normalized input to bilinear
            x_normed = residual * scale

            # Bilinear: h = L(x_normed) * R(x_normed)
            u = block.mlp.L(x_normed)
            v = block.mlp.R(x_normed)
            h = u * v  # [n_patches, hidden_dim]

            # Output: out = D @ h
            D_weight = block.mlp.D.weight  # [embed_dim, hidden_dim]

            # Contribution of channel k at patch p:
            # The channel writes D[:, k] * h[p, k] to the residual
            # This eventually gets normalized by final_rms and projected by W

            # For now, compute raw contribution to residual
            # contrib[p, k, :] = D[:, k] * h[p, k]
            contrib_to_res = h.unsqueeze(-1) * D_weight.T.unsqueeze(0)  # [n_patches, hidden, embed]

            all_contribs.append({
                'h': h,
                'contrib_to_res': contrib_to_res,
                'rms': rms,
            })

            # Update residual
            out = block.mlp.D(h)
            residual = residual + out

        # Now compute how each channel contributes to the final logit
        # Final processing: x_final = residual * (final_gamma / final_rms)
        # logit = W @ mean(x_final) + bias

        final_gamma = model.final_norm.gamma
        final_scale = final_gamma / final_rms.squeeze(0)  # [n_patches, embed_dim]

        # Contribution to logit from channel (layer, p, k):
        # 1. Channel adds D[:, k] * h[p, k] to residual at patch p
        # 2. This gets scaled by final_scale[p]
        # 3. Then mean-pooled and projected by W

        all_channel_contribs = []

        for layer_idx, info in enumerate(all_contribs):
            h = info['h']  # [n_patches, hidden]
            contrib_to_res = info['contrib_to_res']  # [n_patches, hidden, embed]

            # How this contribution affects final logit:
            # scaled_contrib = contrib_to_res * final_scale.unsqueeze(1)
            # logit_contrib = W @ mean(scaled_contrib)

            # For each (p, k):
            # contrib_to_logit[p, k] = (W * final_scale[p]) @ (D[:, k] * h[p, k]) / n_patches
            #                        = (W * final_scale[p] @ D[:, k]) * h[p, k] / n_patches

            # Vectorized: [n_patches, hidden]
            W_scaled = W_y.unsqueeze(0) * final_scale  # [n_patches, embed]
            WD = W_scaled @ D_weight  # [n_patches, hidden]
            contrib_to_logit = (WD * h) / n_patches  # [n_patches, hidden]

            all_channel_contribs.append(contrib_to_logit.reshape(-1))

        # Base contribution (from patch_embed through final norm)
        # This is what logit would be with no bilinear layers
        base_residual = patch_embed
        base_final = base_residual * final_scale
        base_logit = (W_y @ base_final.mean(dim=0)).item() + bias_y

        channel_contribs = torch.cat(all_channel_contribs)

        return {
            'full_logit': full_logit,
            'base_logit': base_logit,
            'channel_contribs': channel_contribs,
            'all_h': [info['h'] for info in all_contribs],
            'all_rms': all_rms,
            'final_rms': final_rms,
        }


def ablate_channel(model, x, y, layer_idx, patch_idx, channel_idx, device):
    """
    Compute logit with ONE channel zeroed out.
    """
    model.eval()
    x_in = x.unsqueeze(0).to(device)

    with torch.no_grad():
        residual = model.patch_embed(x_in).squeeze(0)

        for li, block in enumerate(model.blocks):
            x_normed = block.norm(residual)
            u = block.mlp.L(x_normed)
            v = block.mlp.R(x_normed)
            h = u * v

            if li == layer_idx:
                h[patch_idx, channel_idx] = 0.0

            out = block.mlp.D(h)
            residual = residual + out

        x_final = model.final_norm(residual)
        x_pooled = x_final.mean(dim=0)
        logit = (model.head.weight[y] @ x_pooled).item()
        if model.head.bias is not None:
            logit += model.head.bias[y].item()

    return logit


def verify_sample(model, x, y, device, top_k=10):
    """Verify attribution for top-K channels."""
    info = compute_attribution_frozen_rms(model, x, y, device)

    contribs = info['channel_contribs']
    full_logit = info['full_logit']
    base_logit = info['base_logit']

    n_patches = 64
    hidden_dim = model.hidden_dim
    n_per_layer = n_patches * hidden_dim

    # Check decomposition accuracy
    reconstructed = base_logit + contribs.sum().item()
    decomp_error = abs(full_logit - reconstructed)

    # Get top-K channels by magnitude
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
        'base_logit': base_logit,
        'decomp_error': decomp_error,
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

    lambdas = [0.0, 10.0]

    print("\n" + "="*80)
    print("RMSNorm Attribution Verification (Frozen RMS)")
    print("="*80)

    for lam in lambdas:
        path = f'cifar10_models/patch_bilinear_rmsnorm_attr_{lam}.pt'
        try:
            model, config = load_model(path)
            model = model.to(device)
        except FileNotFoundError:
            print(f"\nModel not found: {path}")
            continue

        print(f"\n{'='*70}")
        print(f"Lambda = {lam}")
        print('='*70)

        n_samples = 10
        all_layer0_errors = []
        all_layer1_errors = []

        for i in range(n_samples):
            x, y = test_dataset[i]
            results, info = verify_sample(model, x, y, device, top_k=20)

            if i == 0:
                print(f"\nSample {i}: full_logit={info['full_logit']:.3f}, base={info['base_logit']:.3f}")
                print(f"Decomposition error: {info['decomp_error']:.6f}")
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

        # Compare to no-norm model
        print(f"\n(Compare to no-LN model errors: Layer0 ~13.7, Layer1 ~0.0)")


if __name__ == '__main__':
    main()

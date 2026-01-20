"""
2-Layer Patch BilinearMLP with RMSNorm (instead of LayerNorm).

RMSNorm with frozen RMS during attribution should give much better
causal accuracy than LayerNorm because it's just element-wise scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time


class RMSNorm(nn.Module):
    """RMSNorm: simpler than LayerNorm, no mean subtraction."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, return_rms=False):
        # x: [..., dim]
        rms = (x.pow(2).mean(dim=-1, keepdim=True)).sqrt() + self.eps
        normed = x * (self.gamma / rms)
        if return_rms:
            return normed, rms
        return normed


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=48):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, self.n_patches, -1)
        x = self.proj(x)
        x = x + self.pos_embed
        return x


class BilinearMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, use_swish=False):
        super().__init__()
        self.L = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.R = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.D = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.use_swish = use_swish

    def forward(self, x):
        u = self.L(x)
        if self.use_swish:
            u = u * torch.sigmoid(u)  # Swish = x * sigmoid(x)
        v = self.R(x)
        h = u * v
        out = self.D(h)
        return out, h


class BilinearMLPBlock(nn.Module):
    """Block with RMSNorm (post-norm style: norm AFTER residual add)."""
    def __init__(self, embed_dim, hidden_dim, use_swish=False):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.mlp = BilinearMLP(embed_dim, hidden_dim, use_swish=use_swish)

    def forward(self, x, return_rms=False):
        # Post-norm: bilinear first, then norm after residual
        mlp_out, h = self.mlp(x)
        residual = x + mlp_out
        if return_rms:
            out, rms = self.norm(residual, return_rms=True)
            return out, h, rms
        else:
            return self.norm(residual), h


class BilinearPatchNetRMSNorm(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=48, hidden_dim=128, n_layers=2, n_classes=10, use_swish=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_swish = use_swish
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.blocks = nn.ModuleList([
            BilinearMLPBlock(embed_dim, hidden_dim, use_swish=use_swish)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_h=False):
        x = self.patch_embed(x)

        if return_h:
            all_h = []
            for block in self.blocks:
                x, h = block(x)
                all_h.append(h)
            # Post-norm: each block already normalized, no final_norm needed
            x = x.mean(dim=1)
            return self.head(x), all_h
        else:
            for block in self.blocks:
                x, _ = block(x)
            # Post-norm: each block already normalized
            x = x.mean(dim=1)
            return self.head(x)

    def forward_with_rms(self, x):
        """Forward pass that stores RMS values for attribution."""
        x = self.patch_embed(x)

        all_h = []
        all_rms = []
        for block in self.blocks:
            x, h, rms = block(x, return_rms=True)
            all_h.append(h)
            all_rms.append(rms)

        # Post-norm: last block already normalized, no final_norm
        x_pooled = x.mean(dim=1)
        logits = self.head(x_pooled)

        return logits, all_h, all_rms

    def forward_with_grads(self, x, y, freeze_rms=True):
        """
        Forward with gradient-based attribution. Returns logits, all_attr, all_h.

        If freeze_rms=True, RMS values are treated as constants during attribution
        (gradients don't flow through RMS calculation). This matches inference-time
        attribution and gives more accurate causal attribution.
        """
        B = x.shape[0]

        if freeze_rms:
            # First pass: collect RMS values
            with torch.no_grad():
                residual = self.patch_embed(x)
                rms_values = []
                for block in self.blocks:
                    mlp_out, h = block.mlp(residual)
                    residual = residual + mlp_out
                    rms = (residual.pow(2).mean(dim=-1, keepdim=True)).sqrt() + block.norm.eps
                    rms_values.append(rms.clone())
                    residual = block.norm(residual)

            # Second pass: forward with frozen RMS for attribution
            residual = self.patch_embed(x)
            all_h = []

            for li, block in enumerate(self.blocks):
                mlp_out, h = block.mlp(residual)
                h.retain_grad()
                all_h.append(h)

                mlp_out = block.mlp.D(h)
                residual = residual + mlp_out

                # Frozen RMS: treat as constant
                frozen_rms = rms_values[li].detach()
                gamma = block.norm.gamma
                residual = residual * (gamma / frozen_rms)

            x_pooled = residual.mean(dim=1)
            logits = self.head(x_pooled)

        else:
            # Original unfrozen version
            x = self.patch_embed(x)
            all_h = []
            for block in self.blocks:
                x, h = block(x)
                h.retain_grad()
                all_h.append(h)
            x_pooled = x.mean(dim=1)
            logits = self.head(x_pooled)

        correct_logits = (logits * F.one_hot(y, logits.shape[1]).float()).sum()
        grads = torch.autograd.grad(correct_logits, all_h, create_graph=True)

        all_attr = []
        for h, grad_h in zip(all_h, grads):
            attr = grad_h * h
            attr_flat = attr.reshape(B, -1)
            all_attr.append(attr_flat)

        return logits, all_attr, all_h


def fast_augment(x):
    B = x.shape[0]
    flip_mask = torch.rand(B, 1, 1, 1, device=x.device) > 0.5
    x = torch.where(flip_mask, x.flip(-1), x)
    x = F.pad(x, (4, 4, 4, 4), mode='reflect')
    sh = torch.randint(0, 9, (1,)).item()
    sw = torch.randint(0, 9, (1,)).item()
    x = x[:, :, sh:sh+32, sw:sw+32]
    return x


def compute_hoyer_per_sample(x):
    x_abs = x.abs()
    n = x.shape[1]
    l1 = x_abs.sum(dim=1)
    l2 = (x_abs ** 2).sum(dim=1).sqrt()
    hoyer = (n ** 0.5 - l1 / (l2 + 1e-10)) / (n ** 0.5 - 1)
    return hoyer


def compute_attribution_hoyer_loss(all_attr):
    hoyers = []
    for attr in all_attr:
        hoyer = compute_hoyer_per_sample(attr)
        hoyers.append(hoyer.mean())
    avg_hoyer = sum(hoyers) / len(hoyers)
    return 1.0 - avg_hoyer, avg_hoyer


def compute_magnitude_loss(all_h):
    """Penalize large h magnitudes to keep RMS stable upon ablation."""
    total = 0
    for h in all_h:
        # Mean squared activation per sample
        total += (h ** 2).mean()
    return total / len(all_h)


def compute_delta_loss(model, all_h, tau_delta=0.1):
    """
    Penalize ||D[:,k]|| * |h_k| exceeding threshold.

    This directly bounds the second-order error when ablating channels.
    δ_k = ||D[:,k]|| * |h_k| controls how much ablating channel k
    perturbs the residual stream and thus the RMS normalization.
    """
    total = 0
    for layer_idx, h in enumerate(all_h):
        # h: [batch, n_patches, hidden_dim] or [n_patches, hidden_dim]
        D = model.blocks[layer_idx].mlp.D.weight  # [embed_dim, hidden_dim]

        # ||D[:,k]|| for each channel k
        D_col_norms = D.norm(dim=0)  # [hidden_dim]

        # δ = ||D[:,k]|| * |h[..., k]|
        # h could be [B, n_patches, hidden] or [n_patches, hidden]
        delta = D_col_norms * h.abs()  # broadcasts to h shape

        # Penalize exceeding threshold (soft hinge loss)
        excess = torch.relu(delta - tau_delta)
        total += excess.pow(2).mean()

    return total


def get_lambda_schedule(epoch, total_epochs, target_lambda, warmup_frac=0.2, ramp_frac=0.5):
    warmup_end = warmup_frac * total_epochs
    ramp_end = ramp_frac * total_epochs
    if epoch < warmup_end:
        return 0.0
    elif epoch < ramp_end:
        progress = (epoch - warmup_end) / (ramp_end - warmup_end)
        return target_lambda * progress
    else:
        return target_lambda


def train(attr_lambda=0.0, delta_lambda=0.0, tau_delta=0.1, use_warmup=True, use_swish=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    embed_dim = 48
    hidden_dim = 128
    n_layers = 2
    batch_size = 128
    epochs = 200
    lr = 1e-3
    weight_decay = 0.05

    print(f"\n2-Layer Patch BilinearMLP with RMSNorm (post-norm)")
    print(f"embed_dim={embed_dim}, hidden_dim={hidden_dim}, use_swish={use_swish}")
    print(f"attr_lambda={attr_lambda}, delta_lambda={delta_lambda}, tau_delta={tau_delta}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = BilinearPatchNetRMSNorm(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_swish=use_swish
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_ce_loss = 0
        total_attr_loss = 0
        n_batches = 0

        if use_warmup:
            current_lambda = get_lambda_schedule(epoch, epochs, attr_lambda)
        else:
            current_lambda = attr_lambda

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = fast_augment(x)

            optimizer.zero_grad()

            if current_lambda > 0 or delta_lambda > 0:
                logits, all_attr, all_h = model.forward_with_grads(x, y)
                ce_loss = F.cross_entropy(logits, y)

                if current_lambda > 0:
                    attr_loss, _ = compute_attribution_hoyer_loss(all_attr)
                else:
                    attr_loss = torch.tensor(0.0)

                if delta_lambda > 0:
                    delta_loss = compute_delta_loss(model, all_h, tau_delta)
                else:
                    delta_loss = torch.tensor(0.0)

                loss = ce_loss + current_lambda * attr_loss + delta_lambda * delta_loss
            else:
                logits = model(x)
                ce_loss = F.cross_entropy(logits, y)
                attr_loss = torch.tensor(0.0)
                delta_loss = torch.tensor(0.0)
                loss = ce_loss

            loss.backward()
            optimizer.step()

            total_ce_loss += ce_loss.item()
            total_attr_loss += attr_loss.item() if isinstance(attr_loss, torch.Tensor) else attr_loss
            n_batches += 1

        scheduler.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    correct += (logits.argmax(-1) == y).sum().item()
                    total += y.size(0)
                acc = 100 * correct / total
                if acc > best_acc:
                    best_acc = acc

            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}: CE={total_ce_loss/n_batches:.4f}, acc={acc:.2f}%, best={best_acc:.2f}%, λ={current_lambda:.3f}")

    print(f"\nFinal: best={best_acc:.2f}%")

    import os
    os.makedirs('cifar10_models', exist_ok=True)
    swish_str = "_swish" if use_swish else ""
    save_path = f'cifar10_models/patch_bilinear_rmsnorm_attr{attr_lambda}_delta{delta_lambda}{swish_str}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'attr_lambda': attr_lambda,
            'delta_lambda': delta_lambda,
            'tau_delta': tau_delta,
            'use_swish': use_swish,
            'norm_type': 'rmsnorm_postnorm',
        },
        'best_acc': best_acc,
    }, save_path)
    print(f"Saved to {save_path}")

    return model, best_acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr_lambda', type=float, default=0.0)
    parser.add_argument('--delta_lambda', type=float, default=0.0)
    parser.add_argument('--tau_delta', type=float, default=0.1)
    parser.add_argument('--use_swish', action='store_true', help='Add Swish after L branch')
    args = parser.parse_args()

    train(attr_lambda=args.attr_lambda, delta_lambda=args.delta_lambda,
          tau_delta=args.tau_delta, use_swish=args.use_swish)

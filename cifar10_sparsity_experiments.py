"""
CIFAR-10 BilinearMixerNet Sparsity Experiments
- No RMSNorm, hidden=64 for both mixer and layer

Experiments:
1. Baseline (no sparsity)
2. Hoyer sparsity on hidden activations (λ = 0.01, 0.1, 1.0)
3. Weight L1 sparsity (λ = 0.001, 0.01, 0.1)
4. Iterative Magnitude Pruning - IMP (target = 50%, 70%, 90%)
5. One-shot Pruning + Fine-tuning (target = 50%, 70%, 90%)

Run overnight with: nohup python cifar10_sparsity_experiments.py > sparsity_exp.log 2>&1 &
Or with Claude: claude --dangerously-skip-permissions --continue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Fixed config: no RMSNorm, h=64 for both
MIXER_HIDDEN = 64
LAYER_HIDDEN = 64
USE_NORM = False

os.makedirs('cifar10_models', exist_ok=True)
os.makedirs('cifar10_images', exist_ok=True)


class BilinearMixer(nn.Module):
    """Token mixing via bilinear operation."""
    def __init__(self, n_patches, hidden_dim, embed_dim):
        super().__init__()
        self.L = nn.Linear(n_patches, hidden_dim, bias=False)
        self.R = nn.Linear(n_patches, hidden_dim, bias=False)
        self.D = nn.Linear(hidden_dim, n_patches, bias=False)
        nn.init.normal_(self.L.weight, std=0.02)
        nn.init.normal_(self.R.weight, std=0.02)
        nn.init.normal_(self.D.weight, std=0.02)

    def forward(self, x):
        # x: [B, n_patches, embed_dim]
        y = x.transpose(1, 2)  # [B, embed_dim, n_patches]
        u = self.L(y)  # [B, embed_dim, hidden_dim]
        v = self.R(y)  # [B, embed_dim, hidden_dim]
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
        nn.init.normal_(self.L.weight, std=0.02)
        nn.init.normal_(self.R.weight, std=0.02)
        nn.init.normal_(self.D.weight, std=0.02)

    def forward(self, x):
        # x: [B, n_patches, embed_dim]
        u = self.L(x)  # [B, n_patches, hidden_dim]
        v = self.R(x)
        h = u * v
        out = self.D(h)
        return x + out, h  # Return hidden activations too


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


def preload_cifar10():
    """Preload CIFAR-10 into GPU memory."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    base_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=base_transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=base_transform)

    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=2)

    train_x, train_y = next(iter(train_loader))
    test_x, test_y = next(iter(test_loader))

    return train_x.to(DEVICE), train_y.to(DEVICE), test_x.to(DEVICE), test_y.to(DEVICE)


def fast_augment(x):
    """Fast GPU augmentation: random flip + random translate."""
    B = x.shape[0]
    flip_mask = torch.rand(B, 1, 1, 1, device=x.device) > 0.5
    x = torch.where(flip_mask, x.flip(-1), x)
    x = F.pad(x, (4, 4, 4, 4), mode='reflect')
    shifts = torch.randint(0, 9, (2,), device=x.device)
    x = x[:, :, shifts[0]:shifts[0]+32, shifts[1]:shifts[1]+32]
    return x


def hoyer_loss(hidden_list, eps=1e-8):
    """
    Compute Hoyer sparsity loss on concatenated hidden activations (vectorized).
    Hoyer = (sqrt(n) - L1/L2) / (sqrt(n) - 1)
    Loss = 1 - hoyer (so minimizing loss maximizes sparsity).
    """
    # Concatenate all hidden activations
    all_h = []
    for h in hidden_list:
        all_h.append(h.reshape(h.shape[0], -1))
    h_cat = torch.cat(all_h, dim=1)  # [B, total_hidden]

    # Vectorized Hoyer computation over batch
    B, n = h_cat.shape
    l1 = h_cat.abs().sum(dim=1)  # [B]
    l2 = torch.sqrt((h_cat ** 2).sum(dim=1) + eps)  # [B]
    sqrt_n = np.sqrt(n)
    hoyer_per_sample = (sqrt_n - l1 / (l2 + eps)) / (sqrt_n - 1 + eps)  # [B]
    avg_hoyer = hoyer_per_sample.mean()

    # Loss = 1 - hoyer (want to maximize hoyer)
    return 1 - avg_hoyer, avg_hoyer


def weight_l1_loss(model):
    """
    L1 penalty on weights as proxy for L0 sparsity.
    """
    total_l1 = 0
    n_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name and ('L.' in name or 'R.' in name or 'D.' in name):
            total_l1 += param.abs().sum()
            n_params += param.numel()
    return total_l1 / n_params  # Normalize by number of params


def weight_sparsity_ratio(model, threshold=1e-3):
    """Compute fraction of near-zero weights."""
    total = 0
    sparse = 0
    for name, param in model.named_parameters():
        if 'weight' in name and ('L.' in name or 'R.' in name or 'D.' in name):
            total += param.numel()
            sparse += (param.abs() < threshold).sum().item()
    return sparse / total if total > 0 else 0


def get_weight_masks(model):
    """Get current weight masks (1 where weight != 0)."""
    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name and ('L.' in name or 'R.' in name or 'D.' in name):
            masks[name] = (param.data != 0).float()
    return masks


def apply_masks(model, masks):
    """Zero out weights according to masks."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data *= masks[name]


def prune_weights_global(model, prune_ratio, existing_masks=None):
    """
    Global magnitude pruning: prune smallest prune_ratio of REMAINING (non-zero) weights.
    Returns updated masks dict.
    """
    # Collect all NON-ZERO weight magnitudes
    nonzero_weights = []

    for name, param in model.named_parameters():
        if 'weight' in name and ('L.' in name or 'R.' in name or 'D.' in name):
            flat = param.data.abs().flatten()
            # Only consider non-zero weights
            nonzero_vals = flat[flat > 0]
            nonzero_weights.append(nonzero_vals)

    if len(nonzero_weights) == 0:
        return existing_masks if existing_masks else {}

    all_nonzero = torch.cat(nonzero_weights)
    if len(all_nonzero) == 0:
        return existing_masks if existing_masks else {}

    # Find threshold based on non-zero weights only
    k = int(len(all_nonzero) * prune_ratio)
    if k == 0:
        return existing_masks if existing_masks else get_weight_masks(model)

    threshold = torch.topk(all_nonzero, k, largest=False).values[-1].item()

    # Create masks (prune weights <= threshold, keep weights > threshold)
    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name and ('L.' in name or 'R.' in name or 'D.' in name):
            masks[name] = (param.data.abs() > threshold).float()

    # Apply masks
    apply_masks(model, masks)

    return masks


def count_nonzero_weights(model):
    """Count non-zero weights in L, R, D layers."""
    total = 0
    nonzero = 0
    for name, param in model.named_parameters():
        if 'weight' in name and ('L.' in name or 'R.' in name or 'D.' in name):
            total += param.numel()
            nonzero += (param.data != 0).sum().item()
    return nonzero, total


@torch.no_grad()
def evaluate_fast(model, test_x, test_y, batch_size=512):
    """Evaluate with preloaded GPU data."""
    model.eval()
    correct = 0
    n_batches = (len(test_x) + batch_size - 1) // batch_size

    for i in range(n_batches):
        x = test_x[i*batch_size:(i+1)*batch_size]
        y = test_y[i*batch_size:(i+1)*batch_size]
        logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()

    return correct / len(test_x)


def train_baseline(model, train_x, train_y, test_x, test_y, epochs=EPOCHS):
    """Train without any sparsity penalty."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_batches = len(train_x) // BATCH_SIZE
    history = {'acc': [], 'loss': []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x), device=DEVICE)
        train_x_s, train_y_s = train_x[perm], train_y[perm]
        total_loss = 0

        for i in range(n_batches):
            x = fast_augment(train_x_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            y = train_y_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        test_acc = evaluate_fast(model, test_x, test_y)
        history['acc'].append(test_acc)
        history['loss'].append(total_loss / n_batches)

        if (epoch + 1) % 10 == 0:
            print(f"  [Baseline] Epoch {epoch+1}: acc={test_acc:.2%}, loss={total_loss/n_batches:.4f}")

    return history


def train_hoyer(model, train_x, train_y, test_x, test_y, lambda_hoyer=0.1, epochs=EPOCHS):
    """Train with Hoyer sparsity penalty on hidden activations."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_batches = len(train_x) // BATCH_SIZE
    history = {'acc': [], 'loss': [], 'hoyer': [], 'ce': []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x), device=DEVICE)
        train_x_s, train_y_s = train_x[perm], train_y[perm]
        total_loss, total_hoyer, total_ce = 0, 0, 0

        for i in range(n_batches):
            x = fast_augment(train_x_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            y = train_y_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            logits, hidden_list = model(x)
            ce_loss = F.cross_entropy(logits, y)
            h_loss, hoyer_val = hoyer_loss(hidden_list)

            loss = ce_loss + lambda_hoyer * h_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_hoyer += hoyer_val.item()
            total_ce += ce_loss.item()

        test_acc = evaluate_fast(model, test_x, test_y)
        history['acc'].append(test_acc)
        history['loss'].append(total_loss / n_batches)
        history['hoyer'].append(total_hoyer / n_batches)
        history['ce'].append(total_ce / n_batches)

        if (epoch + 1) % 10 == 0:
            print(f"  [Hoyer λ={lambda_hoyer}] Epoch {epoch+1}: acc={test_acc:.2%}, "
                  f"hoyer={total_hoyer/n_batches:.4f}, ce={total_ce/n_batches:.4f}")

    return history


def train_weight_sparse(model, train_x, train_y, test_x, test_y, lambda_l1=0.01, epochs=EPOCHS):
    """Train with L1 weight sparsity penalty (proxy for L0)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_batches = len(train_x) // BATCH_SIZE
    history = {'acc': [], 'loss': [], 'sparsity': [], 'ce': []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x), device=DEVICE)
        train_x_s, train_y_s = train_x[perm], train_y[perm]
        total_loss, total_ce = 0, 0

        for i in range(n_batches):
            x = fast_augment(train_x_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            y = train_y_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            logits, _ = model(x)
            ce_loss = F.cross_entropy(logits, y)
            l1_loss = weight_l1_loss(model)

            loss = ce_loss + lambda_l1 * l1_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()

        test_acc = evaluate_fast(model, test_x, test_y)
        sparsity = weight_sparsity_ratio(model)
        history['acc'].append(test_acc)
        history['loss'].append(total_loss / n_batches)
        history['sparsity'].append(sparsity)
        history['ce'].append(total_ce / n_batches)

        if (epoch + 1) % 10 == 0:
            print(f"  [L1 λ={lambda_l1}] Epoch {epoch+1}: acc={test_acc:.2%}, "
                  f"sparsity={sparsity:.2%}, ce={total_ce/n_batches:.4f}")

    return history


def train_iterative_pruning(model, train_x, train_y, test_x, test_y,
                            target_sparsity=0.9, prune_rounds=5, epochs_per_round=30):
    """
    Iterative Magnitude Pruning (IMP):
    1. Train for some epochs
    2. Prune smallest X% of remaining weights
    3. Retrain with pruned weights frozen at 0
    4. Repeat until target sparsity reached
    """
    n_batches = len(train_x) // BATCH_SIZE
    history = {'acc': [], 'sparsity': [], 'ce': []}

    # Calculate pruning schedule (how much to prune each round)
    # After k rounds: remaining = (1 - p)^k = 1 - target_sparsity
    # So p = 1 - (1 - target_sparsity)^(1/k)
    prune_per_round = 1 - (1 - target_sparsity) ** (1 / prune_rounds)

    masks = None  # Will be set after first pruning
    current_sparsity = 0

    for round_idx in range(prune_rounds):
        print(f"    Round {round_idx+1}/{prune_rounds} (target prune this round: {prune_per_round:.1%})")

        # Create fresh optimizer (important for retraining after pruning)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(epochs_per_round):
            model.train()
            perm = torch.randperm(len(train_x), device=DEVICE)
            train_x_s, train_y_s = train_x[perm], train_y[perm]
            total_ce = 0

            for i in range(n_batches):
                x = fast_augment(train_x_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                y = train_y_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Re-apply masks after each step to keep pruned weights at 0
                if masks is not None:
                    apply_masks(model, masks)

                total_ce += loss.item()

            test_acc = evaluate_fast(model, test_x, test_y)
            nonzero, total = count_nonzero_weights(model)
            current_sparsity = 1 - nonzero / total

            history['acc'].append(test_acc)
            history['sparsity'].append(current_sparsity)
            history['ce'].append(total_ce / n_batches)

            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}: acc={test_acc:.2%}, sparsity={current_sparsity:.1%}")

        # Prune after this round (except last round)
        if round_idx < prune_rounds - 1:
            # Prune relative to remaining weights
            nonzero_before, _ = count_nonzero_weights(model)
            masks = prune_weights_global(model, prune_per_round)
            nonzero_after, total = count_nonzero_weights(model)
            print(f"    Pruned: {nonzero_before} -> {nonzero_after} weights "
                  f"(sparsity: {1-nonzero_after/total:.1%})")

    return history


def train_hoyer_plus_imp(model, train_x, train_y, test_x, test_y,
                         lambda_hoyer=0.1, target_sparsity=0.9,
                         prune_rounds=5, epochs_per_round=30):
    """
    Combined Hoyer activation sparsity + Iterative Magnitude Pruning.
    - Hoyer loss encourages sparse activations
    - IMP prunes small weights to zero
    """
    n_batches = len(train_x) // BATCH_SIZE
    history = {'acc': [], 'sparsity': [], 'ce': [], 'hoyer': []}

    prune_per_round = 1 - (1 - target_sparsity) ** (1 / prune_rounds)
    masks = None

    for round_idx in range(prune_rounds):
        print(f"    Round {round_idx+1}/{prune_rounds}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(epochs_per_round):
            model.train()
            perm = torch.randperm(len(train_x), device=DEVICE)
            train_x_s, train_y_s = train_x[perm], train_y[perm]
            total_ce, total_hoyer = 0, 0

            for i in range(n_batches):
                x = fast_augment(train_x_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                y = train_y_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

                logits, hidden_list = model(x)
                ce_loss = F.cross_entropy(logits, y)
                h_loss, hoyer_val = hoyer_loss(hidden_list)
                loss = ce_loss + lambda_hoyer * h_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if masks is not None:
                    apply_masks(model, masks)

                total_ce += ce_loss.item()
                total_hoyer += hoyer_val.item()

            test_acc = evaluate_fast(model, test_x, test_y)
            nonzero, total = count_nonzero_weights(model)
            current_sparsity = 1 - nonzero / total

            history['acc'].append(test_acc)
            history['sparsity'].append(current_sparsity)
            history['ce'].append(total_ce / n_batches)
            history['hoyer'].append(total_hoyer / n_batches)

            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}: acc={test_acc:.2%}, sparsity={current_sparsity:.1%}, hoyer={total_hoyer/n_batches:.3f}")

        # Prune after this round (except last round)
        if round_idx < prune_rounds - 1:
            nonzero_before, _ = count_nonzero_weights(model)
            masks = prune_weights_global(model, prune_per_round, masks)
            nonzero_after, total = count_nonzero_weights(model)
            print(f"    Pruned: {nonzero_before} -> {nonzero_after} weights "
                  f"(sparsity: {1-nonzero_after/total:.1%})")

    return history


def train_oneshot_pruning(model, train_x, train_y, test_x, test_y,
                          target_sparsity=0.9, pretrain_epochs=50, finetune_epochs=100):
    """
    One-shot pruning:
    1. Train to convergence
    2. Prune to target sparsity in one shot
    3. Fine-tune remaining weights
    """
    n_batches = len(train_x) // BATCH_SIZE
    history = {'acc': [], 'sparsity': [], 'ce': [], 'phase': []}

    # Phase 1: Pre-training
    print(f"    Phase 1: Pre-training for {pretrain_epochs} epochs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(pretrain_epochs):
        model.train()
        perm = torch.randperm(len(train_x), device=DEVICE)
        train_x_s, train_y_s = train_x[perm], train_y[perm]
        total_ce = 0

        for i in range(n_batches):
            x = fast_augment(train_x_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            y = train_y_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_ce += loss.item()

        test_acc = evaluate_fast(model, test_x, test_y)
        history['acc'].append(test_acc)
        history['sparsity'].append(0.0)
        history['ce'].append(total_ce / n_batches)
        history['phase'].append('pretrain')

        if (epoch + 1) % 10 == 0:
            print(f"      Pretrain Epoch {epoch+1}: acc={test_acc:.2%}")

    # Phase 2: One-shot pruning
    print(f"    Phase 2: Pruning to {target_sparsity:.0%} sparsity")
    masks = prune_weights_global(model, target_sparsity)
    nonzero, total = count_nonzero_weights(model)
    actual_sparsity = 1 - nonzero / total
    print(f"    Pruned to {nonzero}/{total} weights ({actual_sparsity:.1%} sparsity)")

    # Phase 3: Fine-tuning
    print(f"    Phase 3: Fine-tuning for {finetune_epochs} epochs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)

    for epoch in range(finetune_epochs):
        model.train()
        perm = torch.randperm(len(train_x), device=DEVICE)
        train_x_s, train_y_s = train_x[perm], train_y[perm]
        total_ce = 0

        for i in range(n_batches):
            x = fast_augment(train_x_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            y = train_y_s[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Keep pruned weights at 0
            apply_masks(model, masks)
            total_ce += loss.item()

        test_acc = evaluate_fast(model, test_x, test_y)
        nonzero, total = count_nonzero_weights(model)
        current_sparsity = 1 - nonzero / total

        history['acc'].append(test_acc)
        history['sparsity'].append(current_sparsity)
        history['ce'].append(total_ce / n_batches)
        history['phase'].append('finetune')

        if (epoch + 1) % 10 == 0:
            print(f"      Finetune Epoch {epoch+1}: acc={test_acc:.2%}, sparsity={current_sparsity:.1%}")

    return history


def run_all_experiments():
    print(f"Device: {DEVICE}")
    print(f"Config: mixer_hidden={MIXER_HIDDEN}, layer_hidden={LAYER_HIDDEN}, use_norm={USE_NORM}")
    print()

    # Preload data
    print("Preloading data to GPU...")
    train_x, train_y, test_x, test_y = preload_cifar10()
    print(f"Data loaded. GPU memory: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    print()

    results = {}

    # Experiment 1: Baseline
    print("="*60)
    print("Experiment 1: Baseline (no sparsity)")
    print("="*60)
    model_baseline = BilinearMixerNet(
        mixer_hidden_dim=MIXER_HIDDEN,
        layer_hidden_dim=LAYER_HIDDEN,
    ).to(DEVICE)
    t0 = time.time()
    hist_baseline = train_baseline(model_baseline, train_x, train_y, test_x, test_y)
    results['baseline'] = {
        'history': hist_baseline,
        'best_acc': max(hist_baseline['acc']),
        'final_acc': hist_baseline['acc'][-1],
        'time': time.time() - t0,
    }
    torch.save(model_baseline.state_dict(), 'cifar10_models/mixer_baseline_noNorm_h64.pt')
    print(f"Baseline: best={results['baseline']['best_acc']:.2%}, time={results['baseline']['time']:.0f}s")
    print()

    # Experiment 2: Hoyer sparsity on activations (multiple λ values)
    hoyer_lambdas = [0.01, 0.1, 1.0]
    for lam in hoyer_lambdas:
        print("="*60)
        print(f"Experiment 2: Hoyer activation sparsity (λ={lam})")
        print("="*60)
        model_hoyer = BilinearMixerNet(
            mixer_hidden_dim=MIXER_HIDDEN,
            layer_hidden_dim=LAYER_HIDDEN,
        ).to(DEVICE)
        t0 = time.time()
        hist_hoyer = train_hoyer(model_hoyer, train_x, train_y, test_x, test_y, lambda_hoyer=lam)
        key = f'hoyer_{lam}'
        results[key] = {
            'history': hist_hoyer,
            'best_acc': max(hist_hoyer['acc']),
            'final_acc': hist_hoyer['acc'][-1],
            'final_hoyer': hist_hoyer['hoyer'][-1],
            'time': time.time() - t0,
        }
        torch.save(model_hoyer.state_dict(), f'cifar10_models/mixer_hoyer{lam}_noNorm_h64.pt')
        print(f"Hoyer λ={lam}: best={results[key]['best_acc']:.2%}, "
              f"hoyer={results[key]['final_hoyer']:.4f}, time={results[key]['time']:.0f}s")
        print()

    # Experiment 3: Weight L1 sparsity (multiple λ values)
    l1_lambdas = [0.001, 0.01, 0.1]
    for lam in l1_lambdas:
        print("="*60)
        print(f"Experiment 3: Weight L1 sparsity (λ={lam})")
        print("="*60)
        model_l1 = BilinearMixerNet(
            mixer_hidden_dim=MIXER_HIDDEN,
            layer_hidden_dim=LAYER_HIDDEN,
        ).to(DEVICE)
        t0 = time.time()
        hist_l1 = train_weight_sparse(model_l1, train_x, train_y, test_x, test_y, lambda_l1=lam)
        key = f'weight_l1_{lam}'
        results[key] = {
            'history': hist_l1,
            'best_acc': max(hist_l1['acc']),
            'final_acc': hist_l1['acc'][-1],
            'final_sparsity': hist_l1['sparsity'][-1],
            'time': time.time() - t0,
        }
        torch.save(model_l1.state_dict(), f'cifar10_models/mixer_weightL1_{lam}_noNorm_h64.pt')
        print(f"L1 λ={lam}: best={results[key]['best_acc']:.2%}, "
              f"sparsity={results[key]['final_sparsity']:.2%}, time={results[key]['time']:.0f}s")
        print()

    # Experiment 4: Iterative Magnitude Pruning (IMP)
    imp_sparsities = [0.5, 0.7, 0.9]
    for target in imp_sparsities:
        print("="*60)
        print(f"Experiment 4: Iterative Magnitude Pruning (target={target:.0%})")
        print("="*60)
        model_imp = BilinearMixerNet(
            mixer_hidden_dim=MIXER_HIDDEN,
            layer_hidden_dim=LAYER_HIDDEN,
        ).to(DEVICE)
        t0 = time.time()
        hist_imp = train_iterative_pruning(
            model_imp, train_x, train_y, test_x, test_y,
            target_sparsity=target, prune_rounds=5, epochs_per_round=30
        )
        key = f'imp_{int(target*100)}'
        results[key] = {
            'history': hist_imp,
            'best_acc': max(hist_imp['acc']),
            'final_acc': hist_imp['acc'][-1],
            'final_sparsity': hist_imp['sparsity'][-1],
            'time': time.time() - t0,
        }
        torch.save(model_imp.state_dict(), f'cifar10_models/mixer_imp{int(target*100)}_noNorm_h64.pt')
        print(f"IMP {target:.0%}: best={results[key]['best_acc']:.2%}, "
              f"sparsity={results[key]['final_sparsity']:.1%}, time={results[key]['time']:.0f}s")
        print()

    # Experiment 5: One-shot Pruning
    oneshot_sparsities = [0.5, 0.7, 0.9]
    for target in oneshot_sparsities:
        print("="*60)
        print(f"Experiment 5: One-shot Pruning (target={target:.0%})")
        print("="*60)
        model_os = BilinearMixerNet(
            mixer_hidden_dim=MIXER_HIDDEN,
            layer_hidden_dim=LAYER_HIDDEN,
        ).to(DEVICE)
        t0 = time.time()
        hist_os = train_oneshot_pruning(
            model_os, train_x, train_y, test_x, test_y,
            target_sparsity=target, pretrain_epochs=50, finetune_epochs=100
        )
        key = f'oneshot_{int(target*100)}'
        results[key] = {
            'history': hist_os,
            'best_acc': max(hist_os['acc']),
            'final_acc': hist_os['acc'][-1],
            'final_sparsity': hist_os['sparsity'][-1],
            'time': time.time() - t0,
        }
        torch.save(model_os.state_dict(), f'cifar10_models/mixer_oneshot{int(target*100)}_noNorm_h64.pt')
        print(f"One-shot {target:.0%}: best={results[key]['best_acc']:.2%}, "
              f"sparsity={results[key]['final_sparsity']:.1%}, time={results[key]['time']:.0f}s")
        print()

    # Save results
    # Convert histories to be JSON serializable
    def convert_list(lst):
        """Convert list elements to float if numeric, else keep as-is."""
        result = []
        for x in lst:
            if isinstance(x, str):
                result.append(x)
            else:
                try:
                    result.append(float(x))
                except (TypeError, ValueError):
                    result.append(x)
        return result

    save_results = {}
    for k, v in results.items():
        save_results[k] = {
            key: val if not isinstance(val, dict) else {kk: convert_list(vv) if isinstance(vv, list) else vv for kk, vv in val.items()}
            for key, val in v.items()
        }
    with open('cifar10_models/sparsity_experiment_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Plot results
    plot_results(results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, res in results.items():
        extra = ""
        if 'final_hoyer' in res:
            extra = f", hoyer={res['final_hoyer']:.4f}"
        elif 'final_sparsity' in res:
            extra = f", sparsity={res['final_sparsity']:.2%}"
        print(f"  {name}: best_acc={res['best_acc']:.2%}{extra}")

    return results


def plot_results(results):
    """Plot comparison of all experiments."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Accuracy curves - all experiments
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res['history']['acc'], label=f"{name} ({res['best_acc']:.1%})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy over Training')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 2: CE Loss curves
    ax = axes[0, 1]
    for name, res in results.items():
        if 'ce' in res['history']:
            ax.plot(res['history']['ce'], label=name)
        else:
            ax.plot(res['history']['loss'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 3: Hoyer sparsity for activation-sparse models
    ax = axes[0, 2]
    for name, res in results.items():
        if 'hoyer' in res['history']:
            ax.plot(res['history']['hoyer'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hoyer Sparsity')
    ax.set_title('Activation Sparsity (Hoyer)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Weight sparsity for weight-sparse models
    ax = axes[1, 0]
    for name, res in results.items():
        if 'sparsity' in res['history'] and len(res['history']['sparsity']) > 0:
            ax.plot([s*100 for s in res['history']['sparsity']], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Sparsity (%)')
    ax.set_title('Weight Sparsity (L1 / Pruning)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Accuracy vs Sparsity trade-off (final values)
    ax = axes[1, 1]
    hoyer_results = [(name, res) for name, res in results.items() if 'hoyer' in name]
    weight_results = [(name, res) for name, res in results.items()
                      if 'final_sparsity' in res and res['final_sparsity'] > 0]

    # Hoyer experiments (activation sparsity)
    if hoyer_results:
        hoyer_vals = [res['final_hoyer'] for name, res in hoyer_results]
        hoyer_accs = [res['best_acc'] * 100 for name, res in hoyer_results]
        ax.scatter(hoyer_vals, hoyer_accs, c='blue', s=100, marker='o', label='Hoyer (activation)', zorder=5)
        for (name, res), hv, ha in zip(hoyer_results, hoyer_vals, hoyer_accs):
            ax.annotate(name, (hv, ha), textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Weight sparsity experiments
    if weight_results:
        weight_sparsity = [res['final_sparsity'] for name, res in weight_results]
        weight_accs = [res['best_acc'] * 100 for name, res in weight_results]
        ax.scatter(weight_sparsity, weight_accs, c='red', s=100, marker='s', label='Weight sparsity', zorder=5)
        for (name, res), ws, wa in zip(weight_results, weight_sparsity, weight_accs):
            ax.annotate(name, (ws, wa), textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Add baseline
    if 'baseline' in results:
        ax.axhline(y=results['baseline']['best_acc'] * 100, color='green', linestyle='--',
                   label=f"Baseline ({results['baseline']['best_acc']:.1%})")

    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_title('Accuracy vs Sparsity Trade-off')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 6: Comparison bar chart - best accuracies
    ax = axes[1, 2]
    names = list(results.keys())
    accs = [results[n]['best_acc'] * 100 for n in names]
    colors = []
    for n in names:
        if 'baseline' in n:
            colors.append('green')
        elif 'hoyer' in n:
            colors.append('blue')
        elif 'imp' in n:
            colors.append('orange')
        elif 'oneshot' in n:
            colors.append('purple')
        else:
            colors.append('red')

    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Best Test Accuracy (%)')
    ax.set_title('Best Accuracy Comparison')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 0.5, i, f'{acc:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('cifar10_images/sparsity_experiments.png', dpi=150, bbox_inches='tight')
    print("Saved plot to cifar10_images/sparsity_experiments.png")


if __name__ == "__main__":
    run_all_experiments()

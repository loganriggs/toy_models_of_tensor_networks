"""
Minimal Bilinear Residual Network with Hoyer Sparsity Loss

This file contains:
1. BilinearResBlock: x_next = x + D((Lx) ⊙ (Rx))
2. Training with Hoyer sparsity loss on channel activations
3. Direct logit contribution: head.weight @ D.weight for each channel

Self-contained - only requires: torch, torchvision
"""

# =============================================================================
# CONFIG
# =============================================================================
RETRAIN_MODEL = True
MODEL_PATH = "minimal_bilinear_model.pt"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# 1. BILINEAR LAYER
# =============================================================================

class BilinearResBlock(nn.Module):
    """
    Bilinear residual block: x_next = x + D((Lx) ⊙ (Rx))

    L: [hidden_dim, in_dim] - left projection
    R: [hidden_dim, in_dim] - right projection
    D: [in_dim, hidden_dim] - down projection

    The hidden activations h = (Lx) ⊙ (Rx) are what we want sparse.
    """
    def __init__(self, in_dim, hidden_dim, bias=False):
        super().__init__()
        self.L = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.R = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.D = nn.Linear(hidden_dim, in_dim, bias=bias)

        # Small init for stability
        for m in (self.L, self.R, self.D):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        u = self.L(x)       # [B, hidden_dim]
        v = self.R(x)       # [B, hidden_dim]
        h = u * v           # [B, hidden_dim] - element-wise product
        delta = self.D(h)   # [B, in_dim]
        return x + delta, h  # Return residual output and hidden activations


class BilinearResNet(nn.Module):
    """
    Stack of bilinear residual blocks + classification head.
    """
    def __init__(self, in_dim=784, d_model=16, hidden_dim=6, n_blocks=4, n_classes=10):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model, bias=False)
        self.blocks = nn.ModuleList([
            BilinearResBlock(d_model, hidden_dim) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(d_model, n_classes, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, x):
        """Returns logits and list of hidden activations per block."""
        if x.dim() == 4:  # [B, C, H, W] -> [B, C*H*W]
            x = x.flatten(1)
        x = self.embed(x)
        all_h = []
        for block in self.blocks:
            x, h = block(x)
            all_h.append(h)

        logits = self.head(x)
        return logits, all_h


# =============================================================================
# 2. HOYER SPARSITY LOSS
# =============================================================================

def hoyer_sparsity(h_list, eps=1e-8):
    """
    Hoyer sparsity measure on channel activations.

    Hoyer = (sqrt(n) - L1/L2) / (sqrt(n) - 1)

    Returns value in [0, 1]:
      - 0 = perfectly dense (all equal)
      - 1 = perfectly sparse (one-hot)

    We minimize (1 - Hoyer) to encourage sparsity.

    Args:
        h_list: List of [B, hidden_dim] tensors, one per block
    """
    # Concatenate all hidden activations: [B, total_channels]
    h_cat = torch.cat(h_list, dim=1)

    # Take absolute value and normalize per sample
    h_abs = h_cat.abs()
    h_norm = h_abs / (h_abs.sum(dim=1, keepdim=True) + eps)

    # Compute Hoyer per sample
    n = h_norm.shape[1]
    sqrt_n = (n ** 0.5)

    l1 = h_norm.sum(dim=1)                          # [B]
    l2 = (h_norm ** 2).sum(dim=1).sqrt()            # [B]

    hoyer = (sqrt_n - l1 / (l2 + eps)) / (sqrt_n - 1 + eps)  # [B]

    # Loss = 1 - Hoyer (minimize to maximize sparsity)
    return (1 - hoyer).mean()


# =============================================================================
# 3. TRAINING WITH HOYER LOSS
# =============================================================================

def train_with_hoyer(model, train_loader, epochs=10, lr=1e-3, lambda_hoyer=1.0, device='cuda'):
    """
    Train model with cross-entropy + Hoyer sparsity loss.

    Total loss = CE + lambda_hoyer * (1 - Hoyer)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_ce, total_hoyer = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits, h_list = model(x)

            ce_loss = F.cross_entropy(logits, y)
            hoyer_loss = hoyer_sparsity(h_list)

            loss = ce_loss + lambda_hoyer * hoyer_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent explosion
            optimizer.step()

            total_ce += ce_loss.item()
            total_hoyer += hoyer_loss.item()

        n_batches = len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}  CE={total_ce/n_batches:.4f}  Hoyer={total_hoyer/n_batches:.4f}")

    return model


@torch.no_grad()
def evaluate(model, test_loader, device='cuda'):
    """Compute test accuracy."""
    model.eval()
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


# =============================================================================
# 4. DIRECT LOGIT CONTRIBUTION
# =============================================================================

def compute_direct_logit_effects(model):
    """
    Compute direct effect of each hidden channel on each output class.

    For each block b and channel c:
        effect[b, c, class] = head.weight[class] @ D.weight[:, c]

    This shows how much activating channel c in block b
    directly contributes to each class logit.

    Returns:
        effects: List of [n_classes, hidden_dim] tensors, one per block
    """
    effects = []
    head_W = model.head.weight  # [n_classes, in_dim]

    for block in model.blocks:
        D_W = block.D.weight    # [in_dim, hidden_dim]
        # effect[class, channel] = sum_i(head[class, i] * D[i, channel])
        effect = head_W @ D_W   # [n_classes, hidden_dim]
        effects.append(effect.detach().cpu())

    return effects


def print_direct_effects(effects, class_names=None):
    """Pretty print the direct logit effects."""
    n_classes = effects[0].shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    for block_idx, effect in enumerate(effects):
        print(f"\n=== Block {block_idx} Direct Logit Effects ===")
        print(f"Shape: {effect.shape} (classes x channels)")

        # Show which channels most strongly affect each class
        for c in range(n_classes):
            row = effect[c]
            top_val, top_ch = row.abs().max(0)
            sign = '+' if row[top_ch] > 0 else '-'
            print(f"  {class_names[c]:12s}: strongest channel={top_ch.item()} ({sign}{top_val.item():.3f})")


# =============================================================================
# 5. MAIN DEMO
# =============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # FashionMNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    # Create model
    model = BilinearResNet(
        in_dim=784,      # 28x28 flattened
        hidden_dim=6,    # 6 channels per block
        n_blocks=4,      # 4 blocks = 24 total channels
        n_classes=10
    )
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    import os
    if RETRAIN_MODEL or not os.path.exists(MODEL_PATH):
        # Train with Hoyer sparsity
        print("\n--- Training with Hoyer Loss ---")
        model = train_with_hoyer(
            model, train_loader,
            epochs=25,
            lr=1e-3,
            # lambda_hoyer=1.0,  # Sparsity strength
            lambda_hoyer=0.0,  # Sparsity strength
            device=device
        )
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    else:
        print(f"\n--- Loading model from {MODEL_PATH} ---")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)

    # Evaluate
    acc = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {acc:.2%}")

    # Compute and display direct logit effects
    print("\n--- Direct Logit Effects ---")
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    effects = compute_direct_logit_effects(model)
    print_direct_effects(effects, class_names)

    # Show the actual effect matrix for block 0
    print("\n--- Block 0 Effect Matrix (classes x channels) ---")
    print(effects[0].numpy().round(2))

    return model, effects, test_loader, device, class_names


def show_sample_activations(model, test_loader, device, class_names, n_samples=5):
    """Show which channels activate for specific test samples."""


if __name__ == "__main__":
    model, effects, test_loader, device, class_names = main()

    # Show channel activations for first 5 test samples
    print("\n" + "=" * 70)
    print("SAMPLE CHANNEL ACTIVATIONS")
    print("=" * 70)
    n_samples = 5
    
    # Get first batch
    x, y = next(iter(test_loader))
    x, y = x[:n_samples].to(device), y[:n_samples].to(device)

    model.eval()
    with torch.no_grad():
        logits, h_list = model(x)
        preds = logits.argmax(1)

    print('=' * 70)
    print('Channel Activations for Test Samples')
    print('=' * 70)

    for i in range(n_samples):
        true_label = class_names[y[i]]
        pred_label = class_names[preds[i]]
        correct = '✓' if y[i] == preds[i] else '✗'

        print(f'\nSample {i}: True={true_label:10s} Pred={pred_label:10s} {correct}')
        print('-' * 70)

        # Show activations per block
        for block_idx, h in enumerate(h_list):
            h_sample = h[i].cpu()  # [hidden_dim]
            h_abs = h_sample.abs()

            # Find top activated channels
            top_vals, top_idxs = h_abs.topk(min(3, len(h_abs)))

            print(f'  Block {block_idx}: ', end='')
            for val, idx in zip(top_vals, top_idxs):
                sign = '+' if h_sample[idx] > 0 else '-'
                print(f'ch{idx.item()}={sign}{val.item():.2f}  ', end='')
            print()

        # Show full activation vector concatenated
        all_h = torch.cat([h[i].cpu() for h in h_list])
        print(f'  All 24 channels: {all_h.numpy().round(2)}')
        from matplotlib import pyplot as plt
        emb = model.embed.weight.detach().cpu()
        head = model.head.weight.detach().cpu()
        conn = head @ emb

        for i in range(10):
            class_conn = conn[i]
            class_conn = class_conn.reshape(28, 28)
            plt.imshow(class_conn)
            plt.show()



"""
CIFAR-10 Bilinear Layer Comparison

Compares training with 0, 1, 2, 3, 4 bilinear layers on CIFAR-10.
Uses Muon optimizer and large batch size.

Input: 32x32x3 = 3072 dim (flattened)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# =============================================================================
# CONFIG
# =============================================================================
DEBUG = False  # If True, only train 1 epoch
DATASET = 'cifar10'  # 'cifar10' or 'cifar100'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512
EPOCHS = 1 if DEBUG else 100
LR = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization
DROPOUT = 0.1        # Dropout rate
HIDDEN_DIMS = [64, 256]  # Compare two hidden dimensions (256 = 4x of 64)
EMBED_DIM = 128   # Residual stream dimension (projects from 3072 -> this)

# =============================================================================
# MUON OPTIMIZER
# =============================================================================

def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration for orthogonalization."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer from modded-nanogpt.
    Memory efficient, ~1.5x better sample efficiency than Adam.
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                # Orthogonalize momentum buffer for 2D+ params
                if len(grad.shape) >= 2:
                    grad_2d = buf.view(buf.shape[0], -1)
                    orthogonal_grad = zeroth_power_via_newtonschulz5(grad_2d, steps=ns_steps)
                    update = orthogonal_grad.view_as(buf)
                else:
                    update = buf

                p.data.add_(update, alpha=-lr)


# =============================================================================
# BILINEAR LAYER
# =============================================================================

class BilinearResBlock(nn.Module):
    """
    Bilinear residual block: x_next = x + D((Lx) * (Rx))
    """
    def __init__(self, in_dim, hidden_dim, bias=False):
        super().__init__()
        self.L = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.R = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.D = nn.Linear(hidden_dim, in_dim, bias=bias)

        for m in (self.L, self.R, self.D):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.L(x) * self.R(x)  # [B, hidden_dim]
        return x + self.D(h)


class BilinearNet(nn.Module):
    """
    Bilinear network for CIFAR-10.

    Optional embedding projection, then N bilinear blocks, then classification head.
    """
    def __init__(self, in_dim=3072, embed_dim=None, hidden_dim=64, n_blocks=1, n_classes=10, dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if embed_dim is not None:
            self.embed = nn.Linear(in_dim, embed_dim, bias=False)
            nn.init.normal_(self.embed.weight, std=0.02)
            working_dim = embed_dim
        else:
            self.embed = None
            working_dim = in_dim

        self.blocks = nn.ModuleList([
            BilinearResBlock(working_dim, hidden_dim) for _ in range(n_blocks)
        ])

        self.head = nn.Linear(working_dim, n_classes, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

        self.n_blocks = n_blocks

    def forward(self, x):
        # Flatten: [B, 3, 32, 32] -> [B, 3072]
        if x.dim() == 4:
            x = x.flatten(1)

        # Optional embedding
        if self.embed is not None:
            x = self.embed(x)

        # Bilinear blocks with dropout
        for block in self.blocks:
            x = block(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return self.head(x)


# =============================================================================
# DATA
# =============================================================================

def get_cifar_loaders(batch_size=512, dataset='cifar10', augment=True):
    """Load CIFAR-10 or CIFAR-100 with data augmentation."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'cifar10':
        DS = datasets.CIFAR10
        n_classes = 10
    elif dataset == 'cifar100':
        DS = datasets.CIFAR100
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_data = DS('./data', train=True, download=True, transform=train_transform)
    test_data = DS('./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    return train_loader, test_loader, n_classes


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute accuracy on dataset."""
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total


def train_model(n_blocks, train_loader, test_loader, device, epochs=10, lr=1e-3,
                hidden_dim=64, embed_dim=None, n_classes=10, dropout=0.0, weight_decay=0.0):
    """Train a model with given number of bilinear blocks."""

    model = BilinearNet(
        in_dim=3072,  # 32*32*3
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        n_classes=n_classes,
        dropout=dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training with {n_blocks} bilinear blocks (dropout={dropout}, wd={weight_decay})")
    print(f"Parameters: {n_params:,}")
    print(f"{'='*60}")

    # Use AdamW for weight decay support (Muon doesn't have native weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'test_acc': []}

    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1:2d}/{epochs}  Loss={train_loss:.4f}  Test Acc={test_acc:.2%}")

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.1f}s")

    return model, history


# =============================================================================
# MAIN
# =============================================================================

def main():
    import matplotlib.pyplot as plt

    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET}")
    print(f"Debug mode: {DEBUG} (epochs={EPOCHS})")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Hidden dims: {HIDDEN_DIMS}")
    print(f"Embed dim (residual stream): {EMBED_DIM}")

    train_loader, test_loader, n_classes = get_cifar_loaders(BATCH_SIZE, DATASET)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    print(f"Classes: {n_classes}")

    # Colors for different block counts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    block_counts = [0, 1, 2, 3, 4]

    all_results = {}

    for hidden_dim in HIDDEN_DIMS:
        print(f"\n{'#'*60}")
        print(f"# HIDDEN DIM = {hidden_dim}")
        print(f"{'#'*60}")

        results = {}
        for n_blocks in block_counts:
            model, history = train_model(
                n_blocks=n_blocks,
                train_loader=train_loader,
                test_loader=test_loader,
                device=DEVICE,
                epochs=EPOCHS,
                lr=LR,
                hidden_dim=hidden_dim,
                embed_dim=EMBED_DIM,
                n_classes=n_classes
            )
            results[n_blocks] = {
                'final_acc': history['test_acc'][-1],
                'final_loss': history['train_loss'][-1],
                'history': history
            }

        all_results[hidden_dim] = results

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, n_blocks in enumerate(block_counts):
        color = colors[i]

        for hidden_dim in HIDDEN_DIMS:
            history = all_results[hidden_dim][n_blocks]['history']
            epochs_range = range(1, len(history['test_acc']) + 1)

            if hidden_dim == HIDDEN_DIMS[0]:  # First (smaller) hidden dim - solid
                linestyle = '-'
                label = f"{n_blocks} blocks (h={hidden_dim})"
            else:  # Larger hidden dim - dotted
                linestyle = '--'
                label = f"{n_blocks} blocks (h={hidden_dim})"

            ax.plot(epochs_range, [acc * 100 for acc in history['test_acc']],
                    color=color, linestyle=linestyle, label=label, linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'{DATASET.upper()} - Bilinear Blocks Comparison\nSolid: h={HIDDEN_DIMS[0]}, Dashed: h={HIDDEN_DIMS[1]}', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cifar_bilinear_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to cifar_bilinear_comparison.png")
    plt.show()

    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY: Final Test Accuracy on {DATASET.upper()}")
    print("="*60)
    for hidden_dim in HIDDEN_DIMS:
        print(f"\nHidden dim = {hidden_dim}:")
        for n_blocks, res in all_results[hidden_dim].items():
            print(f"  {n_blocks} blocks: {res['final_acc']:.2%}")

    return all_results


if __name__ == "__main__":
    results = main()

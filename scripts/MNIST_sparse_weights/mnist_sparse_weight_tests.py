import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ============================================================================
# Gradient Estimators for Binary Masks
# ============================================================================

class STEFunction(torch.autograd.Function):
    """Straight-Through Estimator: binary forward, identity backward"""
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ReinMaxFunction(torch.autograd.Function):
    """ReinMax: Heun's method for second-order gradient accuracy"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Heun's method: average of forward and backward Euler
        # Forward Euler gradient: grad_output (STE)
        # Backward Euler requires evaluating at the discrete output
        # Approximation: use smoothed version
        sigmoid_grad = torch.sigmoid(input) * (1 - torch.sigmoid(input))
        # ReinMax uses a corrected gradient that approximates Heun's method
        # Simplified version: weighted combination
        return grad_output * (1 + sigmoid_grad)


class DecoupledGumbelSoftmaxFunction(torch.autograd.Function):
    """Decoupled Gumbel-Softmax with separate forward and backward temperatures"""
    @staticmethod
    def forward(ctx, logits, tau_f, tau_b, hard=True):
        # Forward pass: use tau_f for sampling
        y_soft = torch.sigmoid(logits / tau_f)
        
        if hard:
            # Straight-through: hard decision forward
            y_hard = (logits > 0).float()
            # But keep gradient computation based on soft version
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        
        ctx.save_for_backward(logits)
        ctx.tau_b = tau_b
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        tau_b = ctx.tau_b
        
        # Backward pass: use tau_b for gradients (smoother)
        sigmoid_grad = torch.sigmoid(logits / tau_b) * (1 - torch.sigmoid(logits / tau_b)) / tau_b
        return grad_output * sigmoid_grad, None, None, None


# ============================================================================
# Simple 2-Layer MLP
# ============================================================================

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# Masked MLP for Sparse Training
# ============================================================================

class MaskedMLP(nn.Module):
    def __init__(self, base_model, mask_method='ste'):
        super().__init__()
        self.base_model = base_model
        self.mask_method = mask_method
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Initialize mask parameters (logits)
        self.mask_logits_fc1 = nn.Parameter(torch.randn_like(base_model.fc1.weight) * 0.1 + 2.0)
        self.mask_logits_fc2 = nn.Parameter(torch.randn_like(base_model.fc2.weight) * 0.1 + 2.0)
        
        # Gumbel-Softmax temperatures
        self.tau_f = 0.3  # Forward temperature (sharp)
        self.tau_b = 3.0  # Backward temperature (smooth)
    
    def get_masks(self):
        if self.mask_method == 'ste':
            mask_fc1 = STEFunction.apply(self.mask_logits_fc1)
            mask_fc2 = STEFunction.apply(self.mask_logits_fc2)
        elif self.mask_method == 'reinmax':
            mask_fc1 = ReinMaxFunction.apply(self.mask_logits_fc1)
            mask_fc2 = ReinMaxFunction.apply(self.mask_logits_fc2)
        elif self.mask_method == 'gumbel':
            mask_fc1 = DecoupledGumbelSoftmaxFunction.apply(
                self.mask_logits_fc1, self.tau_f, self.tau_b, True)
            mask_fc2 = DecoupledGumbelSoftmaxFunction.apply(
                self.mask_logits_fc2, self.tau_f, self.tau_b, True)
        else:
            raise ValueError(f"Unknown mask method: {self.mask_method}")
        
        return mask_fc1, mask_fc2
    
    def forward(self, x):
        mask_fc1, mask_fc2 = self.get_masks()
        
        x = x.view(x.size(0), -1)
        
        # Apply masked weights
        masked_weight_fc1 = self.base_model.fc1.weight * mask_fc1
        x = F.linear(x, masked_weight_fc1, self.base_model.fc1.bias)
        x = F.relu(x)
        
        masked_weight_fc2 = self.base_model.fc2.weight * mask_fc2
        x = F.linear(x, masked_weight_fc2, self.base_model.fc2.bias)
        
        return x
    
    def get_sparsity(self):
        mask_fc1, mask_fc2 = self.get_masks()
        total_params = mask_fc1.numel() + mask_fc2.numel()
        active_params = mask_fc1.sum().item() + mask_fc2.sum().item()
        return 1.0 - (active_params / total_params)


# ============================================================================
# Training Functions
# ============================================================================

def train_base_model(device='cuda', epochs=10, model_path='teacher_mlp.pt'):
    """Train the base MLP on MNIST or load from checkpoint"""
    import os

    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        model = MLP().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Evaluate on a small batch to show performance
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        acc = 100. * correct / total
        print(f"Loaded model test accuracy: {acc:.2f}%\n")
        return model

    # Train from scratch
    print(f"Training base MLP on MNIST for {epochs} epochs...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}\n")

    return model


def get_one_per_class_dataset(device='cuda'):
    """Get one example from each class (0-9)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Find one example per class
    indices = []
    class_found = [False] * 10
    
    for idx, (_, label) in enumerate(test_dataset):
        if not class_found[label]:
            indices.append(idx)
            class_found[label] = True
        if all(class_found):
            break
    
    subset = Subset(test_dataset, indices)
    loader = DataLoader(subset, batch_size=10, shuffle=False)
    
    # Get the batch
    data, labels = next(iter(loader))
    return data.to(device), labels.to(device)


def train_sparse_mask(teacher_model, data, labels, mask_method, target_sparsity,
                      device='cuda', steps=500, lr=0.01):
    """Train sparse masks for given data using specified method"""

    masked_model = MaskedMLP(teacher_model, mask_method=mask_method).to(device)
    optimizer = torch.optim.Adam([
        masked_model.mask_logits_fc1,
        masked_model.mask_logits_fc2
    ], lr=lr)

    # Get teacher predictions
    with torch.no_grad():
        teacher_logits = teacher_model(data)
        teacher_probs = F.softmax(teacher_logits, dim=1)

    # Calculate sparsity lambda based on target sparsity
    # Higher target sparsity -> stronger penalty
    # Use exponential scaling: lambda = base * exp(k * target_sparsity)
    lambda_sparsity = target_sparsity

    # Track losses over time
    kl_losses = []
    sparsity_losses = []
    total_losses = []
    actual_sparsities = []
    steps_recorded = []

    masked_model.train()
    pbar = tqdm(range(steps), desc=f"{mask_method} @ {target_sparsity:.0%} sparsity", leave=False)

    for step in pbar:
        optimizer.zero_grad()

        # Student predictions
        student_logits = masked_model(data)
        student_log_probs = F.log_softmax(student_logits, dim=1)

        # KL divergence loss (teacher -> student)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        # Sparsity regularization (L1 penalty on mask logits)
        with torch.no_grad():
            threshold = 1e-5
            total_params = masked_model.mask_logits_fc1.numel() + masked_model.mask_logits_fc2.numel()
            # current_sparsity = ((torch.sigmoid(masked_model.mask_logits_fc1) > threshold).sum() + (torch.sigmoid(masked_model.mask_logits_fc2) > threshold).sum() / total_params).item()
            current_sparsity = ((masked_model.mask_logits_fc1 > threshold).sum() + (masked_model.mask_logits_fc2 > threshold).sum() / total_params).item()
        # current_sparsity = masked_model.get_sparsity()
        # sparsity_loss = masked_model.mask_logits_fc1.norm(p=1) + masked_model.mask_logits_fc2.norm(p=1)
        sparsity_loss = torch.sigmoid(masked_model.mask_logits_fc1).sum() + torch.sigmoid(masked_model.mask_logits_fc2).sum()
        # Total loss with variable lambda
        loss = kl_loss + lambda_sparsity * sparsity_loss

        loss.backward()
        optimizer.step()

        # Record losses every 10 steps
        if step % 10 == 0:
            kl_losses.append(kl_loss.item())
            sparsity_losses.append(sparsity_loss.item())
            total_losses.append(loss.item())
            actual_sparsities.append(current_sparsity)
            steps_recorded.append(step)

        if step % 50 == 0:
            pbar.set_postfix({
                'kl': f'{kl_loss.item():.4f}',
                'sparsity': f'{current_sparsity:.2%}',
                'lambda': f'{lambda_sparsity:.1e}'
            })

    # Evaluate final metrics
    masked_model.eval()
    with torch.no_grad():
        student_logits = masked_model(data)
        student_log_probs = F.log_softmax(student_logits, dim=1)

        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean').item()
        ce_loss = F.cross_entropy(student_logits, labels).item()
        actual_sparsity = masked_model.get_sparsity()

    # Return final metrics and training history
    history = {
        'steps': steps_recorded,
        'kl_losses': kl_losses,
        'sparsity_losses': sparsity_losses,
        'total_losses': total_losses,
        'actual_sparsities': actual_sparsities,
        'lambda': lambda_sparsity
    }

    return kl_div, ce_loss, actual_sparsity, history


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Train base model (or load if exists)
    teacher_model = train_base_model(device=device, epochs=10, model_path='teacher_mlp.pt')
    teacher_model.eval()
    
    # Get one example per class
    print("\nLoading one example per class...")
    data, labels = get_one_per_class_dataset(device=device)
    print(f"Data shape: {data.shape}, Labels: {labels.tolist()}\n")
    
    # Verify teacher performance
    with torch.no_grad():
        teacher_logits = teacher_model(data)
        teacher_preds = teacher_logits.argmax(dim=1)
        teacher_acc = (teacher_preds == labels).float().mean().item()
    print(f"Teacher accuracy on test samples: {teacher_acc*100:.1f}%\n")
    
    # Run experiments across sparsity levels
    methods = ['ste', 'reinmax', 'gumbel']
    sparsity_levels = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    results = {method: {'kl': [], 'ce': [], 'sparsity': [], 'histories': []} for method in methods}

    print("Training sparse masks...\n")
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print('='*60)

        for target_sparsity in sparsity_levels:
            kl_div, ce_loss, actual_sparsity, history = train_sparse_mask(
                teacher_model, data, labels, method, target_sparsity,
                device=device, steps=500, lr=0.02
            )

            results[method]['kl'].append(kl_div)
            results[method]['ce'].append(ce_loss)
            results[method]['sparsity'].append(actual_sparsity)
            results[method]['histories'].append(history)

            print(f"Target: {target_sparsity:.0%} | Actual: {actual_sparsity:.2%} | "
                  f"KL: {kl_div:.4f} | CE: {ce_loss:.4f} | Lambda: {history['lambda']:.2e}")
    
    # Plot results
    print("\nGenerating plots...")
    import os
    figures_dir = 'figures'
    os.makedirs(figures_dir, exist_ok=True)

    colors = {'ste': 'blue', 'reinmax': 'red', 'gumbel': 'green'}
    labels_map = {
        'ste': 'STE (Straight-Through)',
        'reinmax': 'ReinMax (Heun\'s Method)',
        'gumbel': 'Decoupled Gumbel-Softmax'
    }

    # ========================================================================
    # Plot 1: Final Performance vs Sparsity
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # KL Divergence vs Sparsity
    for method in methods:
        sparsities = [s * 100 for s in results[method]['sparsity']]
        ax1.plot(sparsities, results[method]['kl'],
                marker='o', label=labels_map[method], color=colors[method], linewidth=2)

    ax1.set_xlabel('Sparsity (%)', fontsize=12)
    ax1.set_ylabel('KL Divergence (nats)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('KL Divergence vs Sparsity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Cross-Entropy Loss vs Sparsity
    for method in methods:
        sparsities = [s * 100 for s in results[method]['sparsity']]
        ax2.plot(sparsities, results[method]['ce'],
                marker='s', label=labels_map[method], color=colors[method], linewidth=2)

    ax2.set_xlabel('Sparsity (%)', fontsize=12)
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Cross-Entropy Loss vs Sparsity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/binary_mask_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{figures_dir}/binary_mask_comparison.png'")
    plt.close()

    # ========================================================================
    # Plot 2: Training Dynamics - KL Loss over Time
    # ========================================================================
    n_sparsity_levels = len(sparsity_levels)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, target_sparsity in enumerate(sparsity_levels):
        ax = axes[idx]
        for method in methods:
            history = results[method]['histories'][idx]
            ax.plot(history['steps'], history['kl_losses'],
                   label=labels_map[method], color=colors[method], linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Step', fontsize=9)
        ax.set_ylabel('KL Divergence', fontsize=9)
        ax.set_title(f'λ={sparsity_levels[idx]:.1e}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('KL Divergence During Training', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/kl_loss_training_dynamics.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{figures_dir}/kl_loss_training_dynamics.png'")
    plt.close()

    # ========================================================================
    # Plot 3: Training Dynamics - Sparsity Loss over Time
    # ========================================================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, target_sparsity in enumerate(sparsity_levels):
        ax = axes[idx]
        for method in methods:
            history = results[method]['histories'][idx]
            ax.plot(history['steps'], history['sparsity_losses'],
                   label=labels_map[method], color=colors[method], linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Step', fontsize=9)
        ax.set_ylabel('Sparsity Loss (L1 Norm)', fontsize=9)
        ax.set_title(f'λ={sparsity_levels[idx]:.1e}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sparsity Loss During Training', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/sparsity_loss_training_dynamics.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{figures_dir}/sparsity_loss_training_dynamics.png'")
    plt.close()

    # ========================================================================
    # Plot 4: Training Dynamics - Actual Sparsity over Time
    # ========================================================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, target_sparsity in enumerate(sparsity_levels):
        ax = axes[idx]
        for method in methods:
            history = results[method]['histories'][idx]
            sparsity_percentages = [s * 100 for s in history['actual_sparsities']]
            ax.plot(history['steps'], sparsity_percentages,
                   label=labels_map[method], color=colors[method], linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Step', fontsize=9)
        ax.set_ylabel('Actual Sparsity (%)', fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f'λ={sparsity_levels[idx]:.1e}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Actual Sparsity During Training', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/actual_sparsity_training_dynamics.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{figures_dir}/actual_sparsity_training_dynamics.png'")
    plt.close()

    # ========================================================================
    # Plot 5: Combined Loss (KL + Sparsity) over Time
    # ========================================================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, target_sparsity in enumerate(sparsity_levels):
        ax = axes[idx]
        for method in methods:
            history = results[method]['histories'][idx]
            ax.plot(history['steps'], history['total_losses'],
                   label=labels_map[method], color=colors[method], linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Step', fontsize=9)
        ax.set_ylabel('Total Loss', fontsize=9)
        ax.set_title(f'λ={sparsity_levels[idx]:.1e}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Total Loss During Training', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/total_loss_training_dynamics.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{figures_dir}/total_loss_training_dynamics.png'")
    plt.close()
    
    # Print summary table
    print("\n" + "="*95)
    print("SUMMARY TABLE")
    print("="*95)
    print(f"{'Method':<25} {'Target':<10} {'Actual':<10} {'Lambda':<12} {'KL Div':<12} {'CE Loss':<12}")
    print("-"*95)

    for method in methods:
        for i, target in enumerate(sparsity_levels):
            actual = results[method]['sparsity'][i]
            kl = results[method]['kl'][i]
            ce = results[method]['ce'][i]
            lam = results[method]['histories'][i]['lambda']
            print(f"{labels_map[method]:<25} {target*100:>6.0f}% {actual*100:>8.2f}% "
                  f"{lam:>12.2e} {kl:>12.4f} {ce:>12.4f}")
        print("-"*95)


if __name__ == "__main__":
    main()
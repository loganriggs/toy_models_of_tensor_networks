"""
Apply ReinMax binary masking to the bilinear layer of a trained transformer model.
This script loads a pre-trained 3L transformer and applies sparsity constraints
to the first layer's bilinear weights using ReinMax gradient estimation.

Trains separate masks for 10 different samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ToyTransformer, ModelConfig, StreamingTextDataset


# ============================================================================
# ReinMax Gradient Estimator
# ============================================================================

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
        sigmoid_grad = torch.sigmoid(input) * (1 - torch.sigmoid(input))
        # ReinMax uses a corrected gradient
        return grad_output * (1 + sigmoid_grad)

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input > 0).float()

# ============================================================================
# Masked Transformer
# ============================================================================

class MaskedTransformer(nn.Module):
    """Wraps a trained transformer and adds learnable masks to layer 0's bilinear weights"""

    def __init__(self, base_model, mask_layer_idx=0):
        super().__init__()
        self.base_model = base_model
        self.mask_layer_idx = mask_layer_idx

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get the bilinear layer from the specified transformer block
        bilinear = self.base_model.layers[mask_layer_idx].bilinear

        # Initialize mask parameters (logits) for the 3 weight matrices
        # Start with logits = 1.0 (sigmoid(1.0) ≈ 0.731, gives all 1's in forward since > 0)
        # This keeps sigmoid gradient strong enough for optimization
        self.mask_logits_proj1 = nn.Parameter(
            torch.ones_like(bilinear.proj1.weight) * 1.0
        )
        self.mask_logits_proj2 = nn.Parameter(
            torch.ones_like(bilinear.proj2.weight) * 1.0
        )
        self.mask_logits_down = nn.Parameter(
            torch.ones_like(bilinear.down.weight) * 1.0
        )

    def get_masks(self):
        """Apply ReinMax to get binary masks"""
        # mask_proj1 = STEFunction.apply(self.mask_logits_proj1)
        # mask_proj2 = STEFunction.apply(self.mask_logits_proj2)
        # mask_down = STEFunction.apply(self.mask_logits_down)
        mask_proj1 = ReinMaxFunction.apply(self.mask_logits_proj1)
        mask_proj2 = ReinMaxFunction.apply(self.mask_logits_proj2)
        mask_down = ReinMaxFunction.apply(self.mask_logits_down)
        return mask_proj1, mask_proj2, mask_down

    def forward(self, x, attention_mask=None, use_masks=True):
        """Forward pass with masked bilinear layer"""
        # Get masks
        mask_proj1, mask_proj2, mask_down = self.get_masks()

        # Get embeddings
        x_emb = self.base_model.embed(x)
        x_emb = self.base_model.dropout(x_emb)

        # Process through layers
        for i, layer in enumerate(self.base_model.layers):
            if i == self.mask_layer_idx and use_masks:
                # Apply attention normally
                x_emb = x_emb + self.base_model.dropout(layer.attn(x_emb, attention_mask))

                # Apply bilinear with masked weights
                bilinear = layer.bilinear
                masked_proj1_weight = bilinear.proj1.weight * mask_proj1
                masked_proj2_weight = bilinear.proj2.weight * mask_proj2
                masked_down_weight = bilinear.down.weight * mask_down

                # Compute bilinear forward pass manually with masked weights
                hidden1 = F.linear(x_emb, masked_proj1_weight, bilinear.proj1.bias)
                hidden2 = F.linear(x_emb, masked_proj2_weight, bilinear.proj2.bias)
                hidden = hidden1 * hidden2
                output = F.linear(hidden, masked_down_weight, bilinear.down.bias)

                x_emb = x_emb + self.base_model.dropout(output)
            else:
                # Normal forward pass for other layers
                if hasattr(layer, 'attn'):  # TransformerBlock
                    x_emb = layer(x_emb, attention_mask)
                else:  # QuadraticAttention
                    x_emb = layer(x_emb, attention_mask)

        # Output projection
        logits = self.base_model.head(x_emb)

        return logits

    def get_sparsity(self):
        """Calculate the fraction of weights that are zero (sparsity)"""
        mask_proj1, mask_proj2, mask_down = self.get_masks()
        total_params = (mask_proj1.numel() + mask_proj2.numel() +
                       mask_down.numel())
        active_params = (mask_proj1.sum().item() + mask_proj2.sum().item() +
                        mask_down.sum().item())
        return 1.0 - (active_params / total_params)


# ============================================================================
# Training Functions
# ============================================================================

def load_model(model_path, config_path, device='cuda'):
    """Load a trained transformer model"""
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Create model config
    config = ModelConfig(
        vocab_size=config_dict['vocab_size'],
        d_model=config_dict['d_model'],
        n_ctx=config_dict['n_ctx'],
        n_head=config_dict['n_head'],
        dropout=config_dict.get('dropout', 0.1),
        model_type=config_dict['model_type']
    )

    # Create and load model
    model = ToyTransformer(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def get_dataset_samples(dataset_name, tokenizer_name, n_samples=10,
                       seq_length=512, device='cuda', seed=42):
    """Get n separate samples from the dataset with fixed seed for reproducibility"""
    from tokenization.tokenization import enc

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        split='train',
        tokenizer_name=tokenizer_name,
        seq_length=seq_length,
        subset=None,
        validation_ratio=0.001,
        seed=seed
    )

    # Collect samples - each as separate tensor
    samples = []

    for _ in range(n_samples):
        x, _ = dataset.get_batch(batch_size=1, device=device)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(x)

        samples.append((x, attention_mask))

    return samples


def save_trained_masks(masked_model, target_sparsity, sample_idx, output_dir='outputs'):
    """Save trained mask logits to disk"""
    os.makedirs(output_dir, exist_ok=True)

    mask_state = {
        'mask_logits_proj1': masked_model.mask_logits_proj1.detach().cpu(),
        'mask_logits_proj2': masked_model.mask_logits_proj2.detach().cpu(),
        'mask_logits_down': masked_model.mask_logits_down.detach().cpu(),
        'target_sparsity': target_sparsity,
        'sample_idx': sample_idx,
    }

    filename = f'{output_dir}/mask_weights_lambda{target_sparsity:.0e}_sample{sample_idx}.pt'
    torch.save(mask_state, filename)
    return filename


def train_sparse_mask(teacher_model, data, attention_mask, target_sparsity,
                      device='cuda', max_steps=10_000, lr=0.01,
                      early_stop_patience=10_000, use_ce_diff=False):
    """Train sparse masks using ReinMax with early stopping

    Args:
        use_ce_diff: If True, use CE diff as main loss instead of KL divergence
    """

    masked_model = MaskedTransformer(teacher_model, mask_layer_idx=0).to(device)

    # CRITICAL: Disable dropout to match teacher evaluation
    masked_model.base_model.dropout.p = 0.0
    for layer in masked_model.base_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    # Verify masks are initialized to all 1's
    mask_proj1, mask_proj2, mask_down = masked_model.get_masks()
    initial_sparsity = masked_model.get_sparsity()
    masks_all_ones = (
        (mask_proj1 == 1.0).all().item() and
        (mask_proj2 == 1.0).all().item() and
        (mask_down == 1.0).all().item()
    )
    if not masks_all_ones:
        print(f"    WARNING: Masks not all 1's at initialization!")
        print(f"    mask_proj1: {mask_proj1.min().item():.4f} to {mask_proj1.max().item():.4f}")
        print(f"    mask_proj2: {mask_proj2.min().item():.4f} to {mask_proj2.max().item():.4f}")
        print(f"    mask_down: {mask_down.min().item():.4f} to {mask_down.max().item():.4f}")
        print(f"    Initial sparsity: {initial_sparsity:.4f}")

    optimizer = torch.optim.Adam([
        masked_model.mask_logits_proj1,
        masked_model.mask_logits_proj2,
        masked_model.mask_logits_down
    ], lr=lr)

    # Learning rate warmup: ramp up over first 10% of training
    warmup_steps = int(0.05 * max_steps)

    def get_threshold_sparsity(masked_model, threshold=1e-6):
        total_params = (
            masked_model.mask_logits_proj1.numel() +
            masked_model.mask_logits_proj2.numel() +
            masked_model.mask_logits_down.numel()
        )
        active_params = (masked_model.mask_logits_proj1 > threshold).sum().item() + (masked_model.mask_logits_proj2 > threshold).sum().item() + (masked_model.mask_logits_down > threshold).sum().item()
        return 1.0 - (active_params / total_params)
    def get_sparsity_loss(masked_model):
        total_params = (
            masked_model.mask_logits_proj1.numel() +
            masked_model.mask_logits_proj2.numel() +
            masked_model.mask_logits_down.numel()
        )
        # Sparsity regularization (normalized)
        # sparsity_loss = (
        #     masked_model.mask_logits_proj1.abs().sum() +
        #     masked_model.mask_logits_proj2.abs().sum() +
        #     masked_model.mask_logits_down.abs().sum()
        # ) / total_params
        # # Sparsity regularization (normalized)
        sparsity_loss = (
            torch.sigmoid(masked_model.mask_logits_proj1).sum() +
            torch.sigmoid(masked_model.mask_logits_proj2).sum() +
            torch.sigmoid(masked_model.mask_logits_down).sum()
        ) / total_params

        return sparsity_loss


    # Get teacher predictions (teacher is already in eval mode)
    with torch.no_grad():
        teacher_logits = teacher_model(data, attention_mask)[0]
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Calculate teacher CE loss for comparison
        teacher_targets = data[:, 1:]
        teacher_logit_predictions = teacher_logits[:, :-1, :]
        if attention_mask is not None:
            teacher_target_mask = attention_mask[:, 1:]
            teacher_ce_loss_raw = F.cross_entropy(
                teacher_logit_predictions.reshape(-1, teacher_logit_predictions.size(-1)),
                teacher_targets.reshape(-1),
                reduction='none'
            )
            teacher_ce = (teacher_ce_loss_raw * teacher_target_mask.reshape(-1)).sum() / teacher_target_mask.sum()
        else:
            teacher_ce = F.cross_entropy(
                teacher_logit_predictions.reshape(-1, teacher_logit_predictions.size(-1)),
                teacher_targets.reshape(-1)
            )
        teacher_ce = teacher_ce.item()

    # Track losses and metrics
    kl_losses = []
    ce_losses = []
    sparsity_losses = []
    total_losses = []
    actual_sparsities = []
    steps_recorded = []

    # Track gradients
    grad_proj1_norms = []
    grad_proj2_norms = []
    grad_down_norms = []

    # Early stopping based on mask changes
    prev_masks = None
    unchanged_steps = 0

    # Check KL divergence at step 0 (before training)
    masked_model.eval()
    with torch.no_grad():
        initial_student_logits = masked_model(data, attention_mask)
        initial_student_log_probs = F.log_softmax(initial_student_logits, dim=-1)
        initial_kl = F.kl_div(initial_student_log_probs, teacher_probs, reduction='batchmean').item()

        # Calculate initial sparsity loss
        initial_sparsity_loss = get_sparsity_loss(masked_model).item()
        # Also compare logits directly
        logit_diff = (teacher_logits - initial_student_logits).abs().max().item()
        logit_mean_diff = (teacher_logits - initial_student_logits).abs().mean().item()

        # Calculate initial CE diff
        initial_ce_diff = 0.0  # Should be ~0 since masks are all 1's

        print(f"    Teacher CE: {teacher_ce:.6f}")
        print(f"    Initial KL loss: {initial_kl:.6f}")
        print(f"    Initial CE diff: {initial_ce_diff:+.6f}")
        print(f"    Initial sparsity loss: {initial_sparsity_loss:.2f}")

        if use_ce_diff:
            initial_main_loss = initial_ce_diff
            print(f"    Training with CE_diff loss")
            print(f"    Initial total loss (λ={target_sparsity:.1e}): {initial_main_loss + target_sparsity * initial_sparsity_loss:.6f}")
        else:
            initial_main_loss = initial_kl
            print(f"    Training with KL loss")
            print(f"    Initial total loss (λ={target_sparsity:.1e}): {initial_main_loss + target_sparsity * initial_sparsity_loss:.6f}")

        if initial_kl > 1e-6 or logit_diff > 1e-5:
            print(f"    WARNING: Initial KL divergence is non-zero!")
            print(f"    Max logit diff: {logit_diff:.2e}")
            print(f"    Mean logit diff: {logit_mean_diff:.2e}")

    masked_model.train()
    loss_type = "CE_diff" if use_ce_diff else "KL"
    pbar = tqdm(range(max_steps), desc=f"λ={target_sparsity:.0e}, loss={loss_type}", leave=False)

    for step in pbar:
        # Apply learning rate warmup
        if step < warmup_steps:
            warmup_factor = (step + 1) / warmup_steps
            current_lr = lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()

        # Student predictions
        student_logits = masked_model(data, attention_mask)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL divergence loss (teacher -> student)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        # Cross-entropy loss
        targets = data[:, 1:]
        logit_predictions = student_logits[:, :-1, :]
        if attention_mask is not None:
            target_mask = attention_mask[:, 1:]
            ce_loss_raw = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            ce_loss = (ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
        else:
            ce_loss = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1)
            )

        # Sparsity regularization (normalized)
        sparsity_loss = get_sparsity_loss(masked_model)

        # Total loss (use CE diff or KL depending on flag)
        if use_ce_diff:
            ce_diff_loss = (ce_loss - teacher_ce).abs()
            main_loss = ce_diff_loss
        else:
            main_loss = kl_loss

        loss = main_loss + target_sparsity * sparsity_loss

        loss.backward()

        # Track gradient norms (before clipping)
        if step % 10 == 0:
            grad_proj1_norm = masked_model.mask_logits_proj1.grad.norm().item() if masked_model.mask_logits_proj1.grad is not None else 0.0
            grad_proj2_norm = masked_model.mask_logits_proj2.grad.norm().item() if masked_model.mask_logits_proj2.grad is not None else 0.0
            grad_down_norm = masked_model.mask_logits_down.grad.norm().item() if masked_model.mask_logits_down.grad is not None else 0.0
            grad_proj1_norms.append(grad_proj1_norm)
            grad_proj2_norms.append(grad_proj2_norm)
            grad_down_norms.append(grad_down_norm)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_([
            masked_model.mask_logits_proj1,
            masked_model.mask_logits_proj2,
            masked_model.mask_logits_down
        ], max_norm=1.0)
        # ], max_norm=0.1)

        optimizer.step()

        # Check for mask changes (early stopping)
        with torch.no_grad():
            mask_proj1, mask_proj2, mask_down = masked_model.get_masks()
            current_masks = (mask_proj1.clone(), mask_proj2.clone(), mask_down.clone())
            
            if prev_masks is not None:
                # Count how many mask elements changed
                mask_change = (
                    (current_masks[0] != prev_masks[0]).sum().item() +
                    (current_masks[1] != prev_masks[1]).sum().item() +
                    (current_masks[2] != prev_masks[2]).sum().item()
                )
                
                if mask_change == 0:
                    unchanged_steps += 1
                    if unchanged_steps >= early_stop_patience:
                        break
                else:
                    unchanged_steps = 0
            
            prev_masks = current_masks

        # Record metrics
        current_sparsity = masked_model.get_sparsity()

        threshold_sparsity = get_threshold_sparsity(masked_model, threshold=1e-4)
        # threshold_sparsity = 5
        if step % 10 == 0:
            kl_losses.append(kl_loss.item())
            ce_losses.append(ce_loss.item())
            sparsity_losses.append(sparsity_loss.item())
            total_losses.append(loss.item())
            actual_sparsities.append(current_sparsity)
            steps_recorded.append(step)

        if step % 50 == 0:
            ce_diff = ce_loss.item() - teacher_ce
            current_lr = optimizer.param_groups[0]['lr']

            # Show different metrics depending on loss type
            if use_ce_diff:
                postfix_dict = {
                    'ce_diff': f'{ce_diff:+.4f}',
                    'kl': f'{kl_loss.item():.4f}',
                    'sp_loss': f'{sparsity_loss.item():.2e}',
                    'sparsity': f'{current_sparsity:.2%}',
                }
            else:
                postfix_dict = {
                    'kl': f'{kl_loss.item():.4f}',
                    'ce_diff': f'{ce_diff:+.4f}',
                    'sp_loss': f'{sparsity_loss.item():.2e}',
                    'sparsity': f'{current_sparsity:.2%}',
                }

            # Show learning rate during warmup
            if step < warmup_steps:
                postfix_dict['lr'] = f'{current_lr:.2e}'
            pbar.set_postfix(postfix_dict)

    # Final evaluation
    masked_model.eval()
    with torch.no_grad():
        student_logits = masked_model(data, attention_mask)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        final_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean').item()

        # CE loss
        targets = data[:, 1:]
        logit_predictions = student_logits[:, :-1, :]
        if attention_mask is not None:
            target_mask = attention_mask[:, 1:]
            ce_loss_raw = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            final_ce = (ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
            final_ce = final_ce.item()
        else:
            final_ce = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1)
            ).item()

        actual_sparsity = masked_model.get_sparsity()

    # Calculate CE diff (positive means masked is worse)
    final_ce_diff = final_ce - teacher_ce

    # Training history
    history = {
        'steps': steps_recorded,
        'kl_losses': kl_losses,
        'ce_losses': ce_losses,
        'sparsity_losses': sparsity_losses,
        'total_losses': total_losses,
        'actual_sparsities': actual_sparsities,
        'grad_proj1_norms': grad_proj1_norms,
        'grad_proj2_norms': grad_proj2_norms,
        'grad_down_norms': grad_down_norms,
        'lambda': target_sparsity
    }

    return masked_model, final_kl, final_ce_diff, actual_sparsity, history


def apply_threshold_and_verify(masked_model, teacher_model, data, attention_mask,
                               threshold=0.0, device='cuda'):
    """Apply threshold to LOGITS and verify by hard-zeroing"""
    masked_model.eval()

    # Get current mask logits (not masks themselves)
    with torch.no_grad():
        # Threshold the LOGITS, not the binary masks
        hard_mask_proj1 = (masked_model.mask_logits_proj1 > threshold).float()
        hard_mask_proj2 = (masked_model.mask_logits_proj2 > threshold).float()
        hard_mask_down = (masked_model.mask_logits_down > threshold).float()

        # Calculate sparsity with hard threshold
        total_params = (hard_mask_proj1.numel() + hard_mask_proj2.numel() +
                       hard_mask_down.numel())
        active_params = (hard_mask_proj1.sum().item() + hard_mask_proj2.sum().item() +
                        hard_mask_down.sum().item())
        threshold_sparsity = 1.0 - (active_params / total_params)

        # Create a new masked model with hard-thresholded masks
        verified_model = MaskedTransformer(teacher_model, mask_layer_idx=0).to(device)
        verified_model.base_model.dropout.p = 0.0  # Disable dropout
        
        # Set logits to extreme values for hard binary masks
        verified_model.mask_logits_proj1.data = torch.where(
            hard_mask_proj1 > 0.5,
            torch.tensor(10.0, device=device),
            torch.tensor(-10.0, device=device)
        )
        verified_model.mask_logits_proj2.data = torch.where(
            hard_mask_proj2 > 0.5,
            torch.tensor(10.0, device=device),
            torch.tensor(-10.0, device=device)
        )
        verified_model.mask_logits_down.data = torch.where(
            hard_mask_down > 0.5,
            torch.tensor(10.0, device=device),
            torch.tensor(-10.0, device=device)
        )
        verified_model.eval()

        # Evaluate verified model
        teacher_logits = teacher_model(data, attention_mask)[0]
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Calculate teacher CE
        teacher_targets = data[:, 1:]
        teacher_logit_predictions = teacher_logits[:, :-1, :]
        if attention_mask is not None:
            teacher_target_mask = attention_mask[:, 1:]
            teacher_ce_loss_raw = F.cross_entropy(
                teacher_logit_predictions.reshape(-1, teacher_logit_predictions.size(-1)),
                teacher_targets.reshape(-1),
                reduction='none'
            )
            teacher_ce = (teacher_ce_loss_raw * teacher_target_mask.reshape(-1)).sum() / teacher_target_mask.sum()
            teacher_ce = teacher_ce.item()
        else:
            teacher_ce = F.cross_entropy(
                teacher_logit_predictions.reshape(-1, teacher_logit_predictions.size(-1)),
                teacher_targets.reshape(-1)
            ).item()

        student_logits = verified_model(data, attention_mask)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        verified_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean').item()

        # CE loss
        targets = data[:, 1:]
        logit_predictions = student_logits[:, :-1, :]
        if attention_mask is not None:
            target_mask = attention_mask[:, 1:]
            ce_loss_raw = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            verified_ce = (ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
            verified_ce = verified_ce.item()
        else:
            verified_ce = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1)
            ).item()

        # Calculate CE diff (positive means masked is worse)
        verified_ce_diff = verified_ce - teacher_ce

    return threshold_sparsity, verified_kl, verified_ce_diff, hard_mask_proj1, hard_mask_proj2, hard_mask_down


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_results(results, sparsity_levels, output_dir='figures'):
    """Generate plots comparing ReinMax performance vs sparsity"""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Final Performance vs Sparsity (averaged over samples)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sparsities = [s * 100 for s in results['mean_sparsity']]

    # KL Divergence with error bars
    ax1.errorbar(sparsities, results['mean_kl'], yerr=results['std_kl'],
                marker='o', label='ReinMax', color='red', linewidth=2,
                capsize=5, capthick=2)
    ax1.set_xlabel('Sparsity (%)', fontsize=12)
    ax1.set_ylabel('KL Divergence (nats)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('KL Divergence vs Sparsity (mean ± std over 10 samples)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Cross-Entropy Diff with error bars
    ax2.errorbar(sparsities, results['mean_ce_diff'], yerr=results['std_ce_diff'],
                marker='s', label='ReinMax', color='red', linewidth=2,
                capsize=5, capthick=2)
    ax2.set_xlabel('Sparsity (%)', fontsize=12)
    ax2.set_ylabel('CE Diff (+ means masked worse)', fontsize=12)
    ax2.set_title('CE Diff vs Sparsity (mean ± std over samples)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/transformer_bilinear_sparsity_comparison.png',
                dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{output_dir}/transformer_bilinear_sparsity_comparison.png'")
    plt.close()

    # Plot 2: Training Dynamics - KL Loss (show sample 0 only for clarity)
    n_sparsity_levels = len(sparsity_levels)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, target_sparsity in enumerate(sparsity_levels):
        ax = axes[idx]
        # Show first sample's training dynamics
        history = results['all_histories'][idx][0]
        ax.plot(history['steps'], history['kl_losses'],
               label='Sample 0', color='red', linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Step', fontsize=9)
        ax.set_ylabel('KL Divergence', fontsize=9)
        ax.set_title(f'λ={sparsity_levels[idx]:.1e}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('KL Divergence During Training (Sample 0)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transformer_kl_training_dynamics.png',
                dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{output_dir}/transformer_kl_training_dynamics.png'")
    plt.close()

    # Plot 3: Training Dynamics - Actual Sparsity (show sample 0 only)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, target_sparsity in enumerate(sparsity_levels):
        ax = axes[idx]
        history = results['all_histories'][idx][0]
        sparsity_percentages = [s * 100 for s in history['actual_sparsities']]
        ax.plot(history['steps'], sparsity_percentages,
               label='Sample 0', color='red', linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Step', fontsize=9)
        ax.set_ylabel('Actual Sparsity (%)', fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f'λ={sparsity_levels[idx]:.1e}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Actual Sparsity During Training (Sample 0)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transformer_sparsity_training_dynamics.png',
                dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{output_dir}/transformer_sparsity_training_dynamics.png'")
    plt.close()

    # Plot 4: Gradient Norms During Training (show sample 0 only)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, target_sparsity in enumerate(sparsity_levels):
        ax = axes[idx]
        history = results['all_histories'][idx][0]

        # Plot gradient norms for all three weight matrices
        ax.plot(history['steps'], history['grad_proj1_norms'],
               label='proj1', color='red', linewidth=2, alpha=0.7)
        ax.plot(history['steps'], history['grad_proj2_norms'],
               label='proj2', color='blue', linewidth=2, alpha=0.7)
        ax.plot(history['steps'], history['grad_down_norms'],
               label='down', color='green', linewidth=2, alpha=0.7)

        ax.set_xlabel('Training Step', fontsize=9)
        ax.set_ylabel('Gradient Norm', fontsize=9)
        ax.set_yscale('log')
        ax.set_title(f'λ={sparsity_levels[idx]:.1e}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Norms During Training (Sample 0)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transformer_gradient_norms.png',
                dpi=150, bbox_inches='tight')
    print(f"Saved plot to '{output_dir}/transformer_gradient_norms.png'")
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train sparse masks on transformer')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: only run first lambda value with 1 sample')
    parser.add_argument('--use-ce-diff', action='store_true',
                       help='Use CE diff as main loss instead of KL divergence')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.debug:
        print("="*80)
        print("DEBUG MODE: Running only first lambda (0.0) with 1 sample")
        print("="*80)

    loss_type = "CE diff" if args.use_ce_diff else "KL divergence"
    print(f"Using device: {device}")
    print(f"Loss type: {loss_type}\n")

    # Load trained model
    print("Loading trained 3L transformer...")
    model_path = 'models/toy_transformer_simplestories_transformer_3L_10000batches.pt'
    config_path = 'configs/simplestories_transformer_3L_10000batches_config.json'
    teacher_model, model_config = load_model(model_path, config_path, device=device)
    print(f"Model loaded: {model_config.model_type}")
    print(f"  d_model={model_config.d_model}, n_head={model_config.n_head}")
    print(f"  vocab_size={model_config.vocab_size}, n_ctx={model_config.n_ctx}")

    # CRITICAL: Disable dropout on teacher model for consistent evaluation
    print("\nDisabling dropout on teacher model...")
    teacher_model.dropout.p = 0.0
    for layer in teacher_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0
    print("Teacher dropout disabled.\n")

    # Load samples from dataset
    n_samples = 20 if not args.debug else 1
    print(f"Loading {n_samples} samples from SimpleStories dataset...")
    samples = get_dataset_samples(
        dataset_name='SimpleStories/SimpleStories',
        tokenizer_name='SimpleStories/SimpleStories-1.25M',
        n_samples=n_samples,
        seq_length=model_config.n_ctx,
        device=device
    )
    print(f"Loaded {len(samples)} samples")
    print(f"Sample shape: {samples[0][0].shape}\n")

    # Verify teacher performance on all samples
    print("Evaluating teacher model on samples...")
    teacher_losses = []
    with torch.no_grad():
        for data, attention_mask in samples:
            _, loss = teacher_model(data, attention_mask)
            teacher_losses.append(loss.item())
    print(f"Teacher loss (mean): {np.mean(teacher_losses):.4f} ± {np.std(teacher_losses):.4f}\n")

    # Run sparsity sweep with larger lambda values
    # if args.debug:
    #     sparsity_levels = [0.0]
    # else:
    sparsity_levels = [1e-1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0]
    sparsity_levels = [100.0]
    # sparsity_levels = [0.5, 50, 500]
    # sparsity_levels = [1e6]
    # sparsity_levels = [500]

    # Store results across all samples
    all_sample_results = {
        'kl': [],           # List of lists: [sparsity_level][sample_idx]
        'ce_diff': [],      # CE diff: positive means masked is worse
        'sparsity': [],
        'verified_kl': [],
        'verified_ce_diff': [],  # Verified CE diff
        'verified_sparsity': [],
        'all_histories': []  # [sparsity_level][sample_idx]
    }

    print(f"Running sparsity sweep with ReinMax on {n_samples} samples...\n")
    print("="*80)

    for target_sparsity in sparsity_levels:
        print(f"\n{'='*80}")
        print(f"Training with λ = {target_sparsity:.1e}")
        print(f"{'='*80}")

        # Store results for this sparsity level across all samples
        kl_results = []
        ce_diff_results = []
        sparsity_results = []
        verified_kl_results = []
        verified_ce_diff_results = []
        verified_sparsity_results = []
        histories = []

        # Train on each sample independently
        for sample_idx, (data, attention_mask) in enumerate(samples):
            print(f"\n  Sample {sample_idx + 1}/{n_samples}:")

            masked_model, kl_div, ce_diff, actual_sparsity, history = train_sparse_mask(
                teacher_model, data, attention_mask, target_sparsity,
                device=device, max_steps=10_000, lr=0.001,
                # device=device, max_steps=5000, lr=0.01,
                # device=device, max_steps=2000, lr=0.01,
                early_stop_patience=10_000,
                use_ce_diff=args.use_ce_diff
            )

            kl_results.append(kl_div)
            ce_diff_results.append(ce_diff)
            sparsity_results.append(actual_sparsity)
            histories.append(history)

            # Save trained masks
            saved_path = save_trained_masks(masked_model, target_sparsity, sample_idx)
            print(f"    Saved masks to: {saved_path}")

            # Apply threshold and verify
            threshold_sparsity, verified_kl, verified_ce_diff, *_ = apply_threshold_and_verify(
                masked_model, teacher_model, data, attention_mask,
                threshold=0.0, device=device
            )

            verified_kl_results.append(verified_kl)
            verified_ce_diff_results.append(verified_ce_diff)
            verified_sparsity_results.append(threshold_sparsity)

            print(f"    Sparsity={actual_sparsity:.2%}, KL={kl_div:.4f}, CE_diff={ce_diff:+.4f}")
            print(f"    Verified: Sparsity={threshold_sparsity:.2%}, "
                  f"KL={verified_kl:.4f}, CE_diff={verified_ce_diff:+.4f}")

        # Store aggregated results for this sparsity level
        all_sample_results['kl'].append(kl_results)
        all_sample_results['ce_diff'].append(ce_diff_results)
        all_sample_results['sparsity'].append(sparsity_results)
        all_sample_results['verified_kl'].append(verified_kl_results)
        all_sample_results['verified_ce_diff'].append(verified_ce_diff_results)
        all_sample_results['verified_sparsity'].append(verified_sparsity_results)
        all_sample_results['all_histories'].append(histories)

        # Print summary for this sparsity level
        print(f"\n  Summary across {n_samples} samples:")
        print(f"    Sparsity:  {np.mean(sparsity_results):.2%} ± {np.std(sparsity_results):.2%}")
        print(f"    KL:        {np.mean(kl_results):.4f} ± {np.std(kl_results):.4f}")
        print(f"    CE_diff:   {np.mean(ce_diff_results):+.4f} ± {np.std(ce_diff_results):.4f}")

    # Compute mean and std across samples for plotting
    results_for_plotting = {
        'mean_kl': [np.mean(kls) for kls in all_sample_results['kl']],
        'std_kl': [np.std(kls) for kls in all_sample_results['kl']],
        'mean_ce_diff': [np.mean(ce_diffs) for ce_diffs in all_sample_results['ce_diff']],
        'std_ce_diff': [np.std(ce_diffs) for ce_diffs in all_sample_results['ce_diff']],
        'mean_sparsity': [np.mean(spars) for spars in all_sample_results['sparsity']],
        'std_sparsity': [np.std(spars) for spars in all_sample_results['sparsity']],
        'mean_verified_kl': [np.mean(vkls) for vkls in all_sample_results['verified_kl']],
        'std_verified_kl': [np.std(vkls) for vkls in all_sample_results['verified_kl']],
        'mean_verified_ce_diff': [np.mean(vce_diffs) for vce_diffs in all_sample_results['verified_ce_diff']],
        'std_verified_ce_diff': [np.std(vce_diffs) for vce_diffs in all_sample_results['verified_ce_diff']],
        'mean_verified_sparsity': [np.mean(vspars) for vspars in all_sample_results['verified_sparsity']],
        'std_verified_sparsity': [np.std(vspars) for vspars in all_sample_results['verified_sparsity']],
        'all_histories': all_sample_results['all_histories']
    }

    # Plot results
    print("\n" + "="*80)
    print("Generating plots...")
    plot_results(results_for_plotting, sparsity_levels)

    # Print summary table
    print("\n" + "="*80)
    print(f"SUMMARY TABLE (Mean ± Std over {n_samples} samples)")
    print("="*80)
    print(f"{'Lambda':<12} {'Sparsity':<20} {'KL Div':<20} {'CE Diff':<20}")
    print("-"*80)

    for i, target in enumerate(sparsity_levels):
        sparsity_mean = np.mean(all_sample_results['sparsity'][i])
        sparsity_std = np.std(all_sample_results['sparsity'][i])
        kl_mean = np.mean(all_sample_results['kl'][i])
        kl_std = np.std(all_sample_results['kl'][i])
        ce_mean = np.mean(all_sample_results['ce_diff'][i])
        ce_std = np.std(all_sample_results['ce_diff'][i])

        print(f"{target:<12.1e} {sparsity_mean*100:>8.2f}±{sparsity_std*100:<8.2f}% "
              f"{kl_mean:>8.4f}±{kl_std:<9.4f} {ce_mean:>8.4f}±{ce_std:<9.4f}")

    print("\n" + "="*80)
    print("VERIFIED RESULTS (Threshold=0.0)")
    print("="*80)
    print(f"{'Lambda':<12} {'Sparsity':<20} {'KL Div':<20} {'CE Diff':<20}")
    print("-"*80)

    for i, target in enumerate(sparsity_levels):
        sparsity_mean = np.mean(all_sample_results['verified_sparsity'][i])
        sparsity_std = np.std(all_sample_results['verified_sparsity'][i])
        kl_mean = np.mean(all_sample_results['verified_kl'][i])
        kl_std = np.std(all_sample_results['verified_kl'][i])
        ce_mean = np.mean(all_sample_results['verified_ce_diff'][i])
        ce_std = np.std(all_sample_results['verified_ce_diff'][i])

        print(f"{target:<12.1e} {sparsity_mean*100:>8.2f}±{sparsity_std*100:<8.2f}% "
              f"{kl_mean:>8.4f}±{kl_std:<9.4f} {ce_mean:>8.4f}±{ce_std:<9.4f}")

    print("\n" + "="*80)
    print("Experiment complete!")


if __name__ == "__main__":
    main()
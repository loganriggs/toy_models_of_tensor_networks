"""
Debug script for masked transformer to investigate KL divergence issues at initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        sigmoid_grad = torch.sigmoid(input) * (1 - torch.sigmoid(input))
        return grad_output * (1 + sigmoid_grad)


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
        # Start with high logits (all 1's in forward pass)
        self.mask_logits_proj1 = nn.Parameter(
            torch.ones_like(bilinear.proj1.weight) * 10.0
        )
        self.mask_logits_proj2 = nn.Parameter(
            torch.ones_like(bilinear.proj2.weight) * 10.0
        )
        self.mask_logits_down = nn.Parameter(
            torch.ones_like(bilinear.down.weight) * 10.0
        )

    def get_masks(self):
        """Apply ReinMax to get binary masks"""
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
# Loading and Setup
# ============================================================================

def load_model(model_path, config_path, device='cuda'):
    """Load a trained transformer model"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = ModelConfig(
        vocab_size=config_dict['vocab_size'],
        d_model=config_dict['d_model'],
        n_ctx=config_dict['n_ctx'],
        n_head=config_dict['n_head'],
        dropout=config_dict.get('dropout', 0.1),
        model_type=config_dict['model_type']
    )

    model = ToyTransformer(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def get_dataset_samples(dataset_name, tokenizer_name, n_samples=1,
                       seq_length=512, device='cuda'):
    """Get n separate samples from the dataset"""
    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        split='train',
        tokenizer_name=tokenizer_name,
        seq_length=seq_length,
        subset=None,
        validation_ratio=0.001,
        seed=42
    )

    samples = []
    for _ in range(n_samples):
        x, _ = dataset.get_batch(batch_size=1, device=device)
        attention_mask = torch.ones_like(x)
        samples.append((x, attention_mask))

    return samples


def check_dropout_status(model, prefix=""):
    """Recursively check dropout status in all modules"""
    print(f"\n{prefix}Dropout Status:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            print(f"  {name}: p={module.p}, training={module.training}")


# ============================================================================
# Debug Main
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*80)
    print("DEBUG MODE: Investigating KL Divergence at Initialization")
    print("="*80)
    print(f"Using device: {device}\n")

    # Load trained model
    print("Loading trained 3L transformer...")
    model_path = 'models/toy_transformer_simplestories_transformer_3L_10000batches.pt'
    config_path = 'configs/simplestories_transformer_3L_10000batches_config.json'
    teacher_model, model_config = load_model(model_path, config_path, device=device)
    print(f"Model loaded: {model_config.model_type}")
    print(f"  d_model={model_config.d_model}, n_head={model_config.n_head}")
    print(f"  vocab_size={model_config.vocab_size}, n_ctx={model_config.n_ctx}")

    # Check teacher model dropout status
    print("\n" + "="*80)
    print("STEP 1: Check Teacher Model Dropout Status")
    print("="*80)
    check_dropout_status(teacher_model, prefix="Teacher ")

    # Load one sample
    print("\n" + "="*80)
    print("STEP 2: Load Test Sample")
    print("="*80)
    print("Loading 1 sample from dataset...")
    samples = get_dataset_samples(
        dataset_name='SimpleStories/SimpleStories',
        tokenizer_name='SimpleStories/SimpleStories-1.25M',
        n_samples=1,
        seq_length=model_config.n_ctx,
        device=device
    )
    data, attention_mask = samples[0]
    print(f"Sample shape: {data.shape}")

    # Create masked model
    print("\n" + "="*80)
    print("STEP 3: Create Masked Model and Check Initialization")
    print("="*80)
    print("Creating MaskedTransformer...")
    masked_model = MaskedTransformer(teacher_model, mask_layer_idx=0).to(device)

    # Check if masks are initialized as 1s
    print("\nChecking mask initialization...")
    mask_proj1, mask_proj2, mask_down = masked_model.get_masks()
    print(f"mask_proj1 shape: {mask_proj1.shape}")
    print(f"  min: {mask_proj1.min().item():.4f}, max: {mask_proj1.max().item():.4f}")
    print(f"  mean: {mask_proj1.mean().item():.4f}, sum: {mask_proj1.sum().item()}/{mask_proj1.numel()}")
    print(f"  all ones: {torch.all(mask_proj1 == 1.0).item()}")

    print(f"\nmask_proj2 shape: {mask_proj2.shape}")
    print(f"  min: {mask_proj2.min().item():.4f}, max: {mask_proj2.max().item():.4f}")
    print(f"  mean: {mask_proj2.mean().item():.4f}, sum: {mask_proj2.sum().item()}/{mask_proj2.numel()}")
    print(f"  all ones: {torch.all(mask_proj2 == 1.0).item()}")

    print(f"\nmask_down shape: {mask_down.shape}")
    print(f"  min: {mask_down.min().item():.4f}, max: {mask_down.max().item():.4f}")
    print(f"  mean: {mask_down.mean().item():.4f}, sum: {mask_down.sum().item()}/{mask_down.numel()}")
    print(f"  all ones: {torch.all(mask_down == 1.0).item()}")

    print(f"\nSparsity: {masked_model.get_sparsity():.4f} (should be 0.0)")

    # Disable dropout explicitly
    print("\n" + "="*80)
    print("STEP 4: Disable Dropout Explicitly")
    print("="*80)
    print("Setting all dropout probabilities to 0.0...")

    # Set dropout for teacher model
    teacher_model.dropout.p = 0.0
    for layer in teacher_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    # Also set for masked model (same underlying model)
    masked_model.base_model.dropout.p = 0.0
    for layer in masked_model.base_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    check_dropout_status(teacher_model, prefix="Teacher (after) ")
    check_dropout_status(masked_model, prefix="Masked (after) ")

    # Compute teacher predictions
    print("\n" + "="*80)
    print("STEP 5: Compute Teacher Predictions")
    print("="*80)
    with torch.no_grad():
        teacher_logits, teacher_loss = teacher_model(data, attention_mask)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
    print(f"Teacher logits shape: {teacher_logits.shape}")
    print(f"Teacher loss: {teacher_loss.item():.6f}")
    print(f"Teacher logits stats:")
    print(f"  min: {teacher_logits.min().item():.4f}, max: {teacher_logits.max().item():.4f}")
    print(f"  mean: {teacher_logits.mean().item():.4f}, std: {teacher_logits.std().item():.4f}")

    # Compute masked model predictions (before training, step 0)
    print("\n" + "="*80)
    print("STEP 6: Compute Masked Model Predictions (Step 0)")
    print("="*80)
    masked_model.eval()
    with torch.no_grad():
        student_logits = masked_model(data, attention_mask, use_masks=True)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

    print(f"Student logits shape: {student_logits.shape}")
    print(f"Student logits stats:")
    print(f"  min: {student_logits.min().item():.4f}, max: {student_logits.max().item():.4f}")
    print(f"  mean: {student_logits.mean().item():.4f}, std: {student_logits.std().item():.4f}")

    # Compute differences
    print("\n" + "="*80)
    print("STEP 7: Compare Teacher vs Masked Model")
    print("="*80)
    logits_diff = (teacher_logits - student_logits).abs()
    print(f"Logits difference:")
    print(f"  min: {logits_diff.min().item():.4e}, max: {logits_diff.max().item():.4e}")
    print(f"  mean: {logits_diff.mean().item():.4e}, std: {logits_diff.std().item():.4e}")

    # Show where the largest differences are
    max_diff_idx = logits_diff.view(-1).argmax()
    batch_idx = max_diff_idx // (logits_diff.shape[1] * logits_diff.shape[2])
    seq_idx = (max_diff_idx % (logits_diff.shape[1] * logits_diff.shape[2])) // logits_diff.shape[2]
    vocab_idx = max_diff_idx % logits_diff.shape[2]
    print(f"Largest difference at: batch={batch_idx}, seq={seq_idx}, vocab={vocab_idx}")
    print(f"  Teacher: {teacher_logits[batch_idx, seq_idx, vocab_idx].item():.6f}")
    print(f"  Student: {student_logits[batch_idx, seq_idx, vocab_idx].item():.6f}")

    # Compute KL divergence
    print("\n" + "="*80)
    print("STEP 8: Compute KL Divergence at Step 0")
    print("="*80)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    print(f"KL Divergence (teacher -> student): {kl_loss.item():.6f}")
    print(f"Expected KL Divergence: ~0.0 (since masks are all 1s)")

    if kl_loss.item() > 1e-5:
        print("\n⚠️  WARNING: KL divergence is NOT close to 0!")
        print("This indicates the masked model is NOT producing the same outputs as the teacher.")
    else:
        print("\n✓ KL divergence is close to 0, as expected.")

    # Compute CE loss for masked model
    print("\n" + "="*80)
    print("STEP 9: Compute Cross-Entropy Loss")
    print("="*80)
    targets = data[:, 1:]
    logit_predictions = student_logits[:, :-1, :]
    target_mask = attention_mask[:, 1:]
    ce_loss_raw = F.cross_entropy(
        logit_predictions.reshape(-1, logit_predictions.size(-1)),
        targets.reshape(-1),
        reduction='none'
    )
    ce_loss = (ce_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()
    print(f"Teacher CE Loss: {teacher_loss.item():.6f}")
    print(f"Student CE Loss: {ce_loss.item():.6f}")
    print(f"CE Loss Difference: {abs(teacher_loss.item() - ce_loss.item()):.6e}")

    # Try with use_masks=False to verify
    print("\n" + "="*80)
    print("STEP 10: Test with use_masks=False")
    print("="*80)
    print("Computing masked model output with use_masks=False...")
    with torch.no_grad():
        student_logits_no_mask = masked_model(data, attention_mask, use_masks=False)
        student_log_probs_no_mask = F.log_softmax(student_logits_no_mask, dim=-1)

    kl_loss_no_mask = F.kl_div(student_log_probs_no_mask, teacher_probs, reduction='batchmean')
    print(f"KL Divergence with use_masks=False: {kl_loss_no_mask.item():.6f}")

    logits_diff_no_mask = (teacher_logits - student_logits_no_mask).abs()
    print(f"Logits difference with use_masks=False:")
    print(f"  max: {logits_diff_no_mask.max().item():.4e}")
    print(f"  mean: {logits_diff_no_mask.mean().item():.4e}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Masks initialized correctly: {torch.all(mask_proj1 == 1.0).item() and torch.all(mask_proj2 == 1.0).item() and torch.all(mask_down == 1.0).item()}")
    print(f"✓ Sparsity at initialization: {masked_model.get_sparsity():.4f}")
    print(f"✓ Dropout disabled: p=0.0 for all layers")
    print(f"✓ KL divergence at step 0: {kl_loss.item():.6f}")
    print(f"✓ Max logit difference: {logits_diff.max().item():.4e}")

    if kl_loss.item() > 1e-5:
        print("\n⚠️  PROBLEM IDENTIFIED:")
        print("The masked model with all-ones masks is NOT matching the teacher model!")
        print("This suggests an issue with the forward pass implementation.")
    else:
        print("\n✓ No issues found - masked model matches teacher at initialization.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()

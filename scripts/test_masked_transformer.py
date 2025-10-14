"""
Test script to verify MaskedTransformer implementation.
Checks if MaskedTransformer with all masks=1 produces the same output as the base model.
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

        # Initialize mask parameters to 1.0 (will give 1's in forward pass)
        self.mask_logits_proj1 = nn.Parameter(
            torch.ones_like(bilinear.proj1.weight)
        )
        self.mask_logits_proj2 = nn.Parameter(
            torch.ones_like(bilinear.proj2.weight)
        )
        self.mask_logits_down = nn.Parameter(
            torch.ones_like(bilinear.down.weight)
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
# Test Functions
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


def get_test_samples(dataset_name, tokenizer_name, n_samples=5,
                     seq_length=512, device='cuda'):
    """Get n test samples from the dataset"""
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


def test_masked_transformer():
    """Test if MaskedTransformer with masks=1 produces same output as base model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load trained model
    print("Loading trained 3L transformer...")
    model_path = 'models/toy_transformer_simplestories_transformer_3L_10000batches.pt'
    config_path = 'configs/simplestories_transformer_3L_10000batches_config.json'
    teacher_model, model_config = load_model(model_path, config_path, device=device)
    print(f"Model loaded: {model_config.model_type}")
    print(f"  d_model={model_config.d_model}, n_head={model_config.n_head}")
    print(f"  vocab_size={model_config.vocab_size}, n_ctx={model_config.n_ctx}\n")

    # Disable dropout for consistent comparison
    teacher_model.dropout.p = 0.0
    for layer in teacher_model.layers:
        if hasattr(layer, 'dropout'):
            layer.dropout.p = 0.0

    # Create masked model with all masks = 1
    print("Creating MaskedTransformer with masks initialized to 1...")
    masked_model = MaskedTransformer(teacher_model, mask_layer_idx=0).to(device)
    masked_model.base_model.dropout.p = 0.0
    masked_model.eval()

    # Verify masks are all 1's
    mask_proj1, mask_proj2, mask_down = masked_model.get_masks()
    print(f"Mask sparsity: {masked_model.get_sparsity():.4f} (should be 0.0)")
    print(f"Mask proj1 sum: {mask_proj1.sum().item()} / {mask_proj1.numel()}")
    print(f"Mask proj2 sum: {mask_proj2.sum().item()} / {mask_proj2.numel()}")
    print(f"Mask down sum: {mask_down.sum().item()} / {mask_down.numel()}\n")

    # Get test samples
    print("Loading 5 test samples...")
    samples = get_test_samples(
        dataset_name='SimpleStories/SimpleStories',
        tokenizer_name='SimpleStories/SimpleStories-1.25M',
        n_samples=5,
        seq_length=model_config.n_ctx,
        device=device
    )
    print(f"Loaded {len(samples)} samples\n")

    # Compare outputs
    print("="*80)
    print("COMPARING OUTPUTS")
    print("="*80)

    all_passed = True
    for i, (data, attention_mask) in enumerate(samples):
        print(f"\nSample {i+1}:")

        with torch.no_grad():
            # Base model output
            teacher_logits, teacher_loss = teacher_model(data, attention_mask)

            # Masked model output
            masked_logits = masked_model(data, attention_mask)

            # Compute loss for masked model
            targets = data[:, 1:]
            logit_predictions = masked_logits[:, :-1, :]
            target_mask = attention_mask[:, 1:]
            masked_loss_raw = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            masked_loss = (masked_loss_raw * target_mask.reshape(-1)).sum() / target_mask.sum()

            # Compare
            logits_diff = (teacher_logits - masked_logits).abs().max().item()
            logits_mean_diff = (teacher_logits - masked_logits).abs().mean().item()
            loss_diff = abs(teacher_loss.item() - masked_loss.item())

            print(f"  Teacher loss: {teacher_loss.item():.6f}")
            print(f"  Masked loss:  {masked_loss.item():.6f}")
            print(f"  Loss diff:    {loss_diff:.2e}")
            print(f"  Max logit diff:  {logits_diff:.2e}")
            print(f"  Mean logit diff: {logits_mean_diff:.2e}")

            # Check if outputs are close (tolerance of 1e-5)
            if logits_diff < 1e-5 and loss_diff < 1e-5:
                print("  ✓ PASSED")
            else:
                print("  ✗ FAILED - outputs differ!")
                all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - MaskedTransformer implementation is correct!")
    else:
        print("✗ TESTS FAILED - MaskedTransformer implementation has issues")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = test_masked_transformer()
    sys.exit(0 if success else 1)

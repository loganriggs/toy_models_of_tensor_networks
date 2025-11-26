import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class BilinearMixerConfig:
    vocab_size: int = None
    d_model: int = 128
    n_ctx: int = 256
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.1
    scale_init: float = 0.02
    causal: bool = True


class BilinearMixerTransformerBlock(nn.Module):
    """
    A single block combining token mixing and feature mixing via bilinear operations.
    
    This serves as both "attention" (token mixing via T matrices) and 
    "MLP" (feature mixing via F matrices) in one unified operation.
    
    Computes: y = (T_l @ x @ F_l^T) * (T_r @ x @ F_r^T)
    
    Where:
    - T_l, T_r are (seq, seq) lower-triangular matrices for causal token mixing
    - F_l, F_r are (d_model, d_model) matrices for feature mixing
    
    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension
        n_heads: Number of bilinear "heads" (rank of the interaction)
        causal: Whether to apply causal masking to token mixing
        scale_init: Initialization scale (smaller = more stable at init)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int = 1,
        causal: bool = True,
        scale_init: float = 0.02,
        dropout: float = 0.0
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal
        
        # Token mixing matrices per head: (n_heads, seq, seq)
        self.T_l = nn.Parameter(torch.empty(n_heads, seq_len, seq_len))
        self.T_r = nn.Parameter(torch.empty(n_heads, seq_len, seq_len))
        
        # Feature mixing matrices per head: (n_heads, d_model, d_model)
        self.F_l = nn.Parameter(torch.empty(n_heads, d_model, d_model))
        self.F_r = nn.Parameter(torch.empty(n_heads, d_model, d_model))
        
        # Causal mask for token mixing
        if causal:
            self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights(scale_init)
    
    def _init_weights(self, scale):
        # Small init for stability - bilinear products can explode
        nn.init.normal_(self.T_l, std=scale)
        nn.init.normal_(self.T_r, std=scale)
        nn.init.normal_(self.F_l, std=scale)
        nn.init.normal_(self.F_r, std=scale)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, S, D) where S <= seq_len
            attention_mask: Optional, not used but kept for interface compatibility
        Returns:
            out: (B, S, D)
        """
        B, S, D = x.shape
        
        # Slice to actual sequence length
        T_l = self.T_l[:, :S, :S]  # (H, S, S)
        T_r = self.T_r[:, :S, :S]  # (H, S, S)
        
        # Apply causal mask to token mixing matrices
        if self.causal:
            mask = self.mask[:S, :S]  # (S, S)
            T_l = T_l * mask
            T_r = T_r * mask
        
        # Compute y_l = T_l @ x @ F_l^T for all heads
        # y_l[b,h,i,k] = sum_j sum_d T_l[h,i,j] * x[b,j,d] * F_l[h,k,d]
        y_l = torch.einsum('hij,bjd,hkd->bhik', T_l, x, self.F_l)  # (B, H, S, D)
        y_r = torch.einsum('hij,bjd,hkd->bhik', T_r, x, self.F_r)  # (B, H, S, D)
        
        # Element-wise product and sum over heads
        y = (y_l * y_r).sum(dim=1)  # (B, S, D)
        
        # Normalize by number of heads to keep scale consistent
        y = y / self.n_heads
        
        return x + y


class BilinearMixerTransformer(nn.Module):
    """
    Full transformer model using BilinearMixerTransformerBlocks.
    
    Structure:
    - Token embedding
    - Stack of BilinearMixerTransformerBlocks
    - Output projection (unembedding)
    """
    
    def __init__(self, config: BilinearMixerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Stack of blocks
        self.layers = nn.ModuleList([
            BilinearMixerTransformerBlock(
                seq_len=config.n_ctx,
                d_model=config.d_model,
                n_heads=config.n_heads,
                causal=config.causal,
                scale_init=config.scale_init,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Output head (unembedding)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize embedding and linear layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass with automatic loss computation.
        
        Args:
            x: Input token ids (B, S)
            attention_mask: Optional mask for padding (B, S)
        
        Returns:
            logits: (B, S, vocab_size)
            loss: Cross-entropy loss
        """
        original_x = x
        
        # Embed tokens
        x = self.embed(x)
        x = self.dropout(x)
        
        # Forward through blocks
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Project to vocabulary
        logits = self.head(x)
        
        # Compute loss (next token prediction)
        targets = original_x[:, 1:]
        logit_predictions = logits[:, :-1, :]
        
        if attention_mask is not None:
            # Only compute loss on non-padded positions
            target_mask = attention_mask[:, 1:]
            loss = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            loss = (loss * target_mask.reshape(-1)).sum() / target_mask.sum()
        else:
            loss = F.cross_entropy(
                logit_predictions.reshape(-1, logit_predictions.size(-1)),
                targets.reshape(-1)
            )
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.
        
        Args:
            idx: Initial context (B, S)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated sequence including initial context
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# Quick test
if __name__ == "__main__":
    torch.manual_seed(42)
    
    config = BilinearMixerConfig(
        vocab_size=1000,
        d_model=64,
        n_ctx=32,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    )
    
    model = BilinearMixerTransformer(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"\nParameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Break down parameters
    print(f"\nParameter breakdown:")
    print(f"  Embedding: {config.vocab_size} × {config.d_model} = {config.vocab_size * config.d_model:,}")
    print(f"  Per block: 2×{config.n_heads}×{config.n_ctx}² + 2×{config.n_heads}×{config.d_model}² = {2*config.n_heads*config.n_ctx**2 + 2*config.n_heads*config.d_model**2:,}")
    print(f"  Head: {config.d_model} × {config.vocab_size} = {config.d_model * config.vocab_size:,}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"\nGeneration test:")
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    
    # Causality check
    print("\nCausality check:")
    x_test = torch.randint(0, config.vocab_size, (1, seq_len))
    logits_test, _ = model(x_test)
    
    x_perturbed = x_test.clone()
    x_perturbed[0, -1] = (x_perturbed[0, -1] + 500) % config.vocab_size
    logits_perturbed, _ = model(x_perturbed)
    
    diff = (logits_test[0, :-1, :] - logits_perturbed[0, :-1, :]).abs().max()
    print(f"  Max change at earlier positions: {diff:.2e}")
    print(f"  Causal: {diff < 1e-5}")
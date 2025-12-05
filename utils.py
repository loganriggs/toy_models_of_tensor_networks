import torch
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.colors as colors
from IPython.display import display, HTML


"""
Toy Transformer Training Script with NanoGPT Speedrun Improvements
Incorporates key improvements from modded-nanogpt for training small experimental models
Now with real data streaming from HuggingFace datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from einops import rearrange, einsum
import math
from dataclasses import dataclass
from typing import Optional, Literal
import time
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, LlamaForCausalLM


# ============================================================================
# Muon Optimizer from modded-nanogpt
# ============================================================================

def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration for orthogonalization used in Muon."""
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
    Muon optimizer from modded-nanogpt
    Memory efficient, ~1.5x better sample efficiency than Adam
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
                
                # Orthogonalize momentum buffer
                if len(grad.shape) >= 2:
                    grad_2d = buf.view(buf.shape[0], -1)
                    orthogonal_grad = zeroth_power_via_newtonschulz5(grad_2d, steps=ns_steps)
                    buf = orthogonal_grad.view_as(buf)
                
                p.data.add_(buf, alpha=-lr)


# ============================================================================
# Core Components
# ============================================================================

class Rotary(nn.Module):
    """Rotary positional embeddings"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache()
        
    def _build_cache(self):
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, :, None, :]
        sin_cached = emb.sin()[None, :, None, :]
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)
        
    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len]
        sin = self.sin_cached[:, :seq_len]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rotated = torch.stack(
            [-x_odd, x_even],
            dim=-1
        ).flatten(-2)
        
        return x * cos + x_rotated * sin


class Component(nn.Module):
    """Base component class"""
    pass


class Mask(nn.Module):
    """Masking for attention patterns"""
    def __init__(self, n_ctx, mask_type='causal'):
        super().__init__()
        self.n_ctx = n_ctx
        self.mask_type = mask_type
        
        if mask_type == 'causal':
            mask = torch.tril(torch.ones(n_ctx, n_ctx))
        else:  # no mask
            mask = torch.ones(n_ctx, n_ctx)
        
        self.register_buffer('mask', mask)
        
    def forward(self, scores):
        seq_len = scores.shape[-1]
        mask = self.mask[:seq_len, :seq_len]
        return scores * mask


class SoftmaxAttention(Component):
    """Standard scaled dot-product attention with softmax using F.scaled_dot_product_attention"""
    def __init__(self, d_model: int, n_head: int, n_ctx: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model

        self.rotary = Rotary(self.d_head, n_ctx)

        # Initialize QKV projections
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)

        # Zero-initialize output projection (muP-like)
        self.o = nn.Linear(d_model, d_model, bias=False)
        init.zeros_(self.o.weight)

    def forward(self, x, attention_mask=None):
        B, S, _ = x.shape
        q, k, v = [rearrange(op(x), 'b s (n_head d_head) -> b n_head s d_head',
                            n_head=self.n_head) for op in [self.q, self.k, self.v]]

        # Apply rotary embeddings
        q = rearrange(q, 'b n_head s d_head -> b s n_head d_head')
        k = rearrange(k, 'b n_head s d_head -> b s n_head d_head')
        q, k = self.rotary(q), self.rotary(k)
        q = rearrange(q, 'b s n_head d_head -> b n_head s d_head')
        k = rearrange(k, 'b s n_head d_head -> b n_head s d_head')

        # Use F.scaled_dot_product_attention with causal mask
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        z = rearrange(z, 'b n_head s d_head -> b s (n_head d_head)')

        return self.o(z)


class QuadraticAttention(Component):
    """Attention using quadratic scoring function instead of softmax"""
    def __init__(self, d_model: int, n_head: int, n_ctx: int,
                 mask: str = 'causal', scale: int = 1,
                 norm: bool = True, bias: bool = True) -> None:
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.scale = scale

        self.rotary = Rotary(self.d_head, n_ctx)
        self.norm = nn.RMSNorm(self.d_head) if norm else nn.Identity()
        self.mask = Mask(n_ctx, mask)

        # Initialize QKV projections
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)

        # Zero-initialize output projection (muP-like)
        self.o = nn.Linear(d_model, d_model, bias=False)
        init.zeros_(self.o.weight)

    def forward(self, x, attention_mask=None):
        q, k, v = [rearrange(op(x), '... (n_head d_head) -> ... n_head d_head',
                            n_head=self.n_head) for op in [self.q, self.k, self.v]]

        # Apply rotary embeddings and normalization
        q, k = self.rotary(self.norm(q)), self.rotary(self.norm(k))

        # Quadratic scoring function
        scores = einsum(q, k, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        pattern = self.mask((scores / self.d_head).square())

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for heads: [batch, seq] -> [batch, 1, seq, seq]
            attn_mask = attention_mask[:, None, None, :]
            # Set masked positions to 0 (since you're using quadratic, not softmax)
            pattern = pattern * attn_mask

        # Aggregate values
        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... seq_q n_head d_head")
        z = rearrange(z, '... seq n_head d_head -> ... seq (n_head d_head)')

        return self.o(z) * self.scale


class ReLUSquared(nn.Module):
    """ReLU² activation from modded-nanogpt"""
    def forward(self, x):
        return F.relu(x).square()


class BilinearLayer(nn.Module):
    """Bilinear layer: element-wise product of two linear projections
    Similar to SwiGLU but without the gating nonlinearity"""
    def __init__(self, d_model, d_hidden=None, bias=False):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model

        # Two parallel projections to hidden dimension
        self.proj1 = nn.Linear(d_model, d_hidden, bias=False)
        self.proj2 = nn.Linear(d_model, d_hidden, bias=False)

        # Output projection
        self.down = nn.Linear(d_hidden, d_model, bias=True)

        # Zero-initialize down projection for stability
        init.zeros_(self.down.weight)
        if bias:
            init.zeros_(self.down.bias)

    def forward(self, x):
        # Compute two parallel projections and multiply element-wise
        hidden = self.proj1(x) * self.proj2(x)
        return self.down(hidden)


class BilinearMixer(nn.Module):
    """
    Bilinear MLP-Mixer for sequence mixing.

    For each position i, computes:
        out[i] = sum over heads of (L[i,:] @ x) * (R[i,:] @ x)

    where L and R are lower-triangular for causal masking.

    With r heads, the interaction matrix at position i has rank r.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        r: int = 1,
        causal: bool = True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.r = r
        self.causal = causal

        # L and R matrices for each head
        self.L = nn.Parameter(torch.empty(r, seq_len, seq_len))
        self.R = nn.Parameter(torch.empty(r, seq_len, seq_len))

        # Output projection to combine heads
        self.out_proj = nn.Linear(d_model * r, d_model, bias=False)

        # Causal mask
        if causal:
            self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))

        self._init_weights()

    def _init_weights(self):
        # Small init to keep outputs stable at start
        nn.init.normal_(self.L, std=0.02)
        nn.init.normal_(self.R, std=0.02)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, S, D) where S <= seq_len
            attention_mask: Optional attention mask (not used, for compatibility)
        Returns:
            out: (B, S, D)
        """
        B, S, D = x.shape

        L, R = self.L[:, :S, :S], self.R[:, :S, :S]

        # Apply causal mask
        if self.causal:
            mask = self.mask[:S, :S]
            L = L * mask
            R = R * mask

        # Bilinear mixing: (B, r, S, D)
        left = torch.einsum('rij,bjd->brid', L, x)
        right = torch.einsum('rij,bjd->brid', R, x)
        mixed = left * right

        # Combine heads: (B, S, r*D) -> (B, S, D)
        mixed = mixed.permute(0, 2, 1, 3).reshape(B, S, self.r * D)
        out = self.out_proj(mixed)

        return out


class TransformerBlock(nn.Module):
    """Single transformer block with attention (quadratic, softmax, bilinear, or mixer) and bilinear layer"""
    def __init__(self, d_model, n_head, n_ctx, dropout=0.0, use_mixer=False, mixer_r=None, use_softmax=False, use_bilinear_attn=False, bilinear_attn_qk_norm=False):
        super().__init__()
        if use_mixer:
            r = mixer_r if mixer_r is not None else n_head
            self.attn = BilinearMixer(seq_len=n_ctx, d_model=d_model, r=r, causal=True)
        elif use_softmax:
            self.attn = SoftmaxAttention(d_model, n_head, n_ctx)
        elif use_bilinear_attn:
            self.attn = CausalBilinearSelfAttention(d_model, n_head, n_ctx, qk_norm=bilinear_attn_qk_norm)
        else:
            self.attn = QuadraticAttention(d_model, n_head, n_ctx)
        self.bilinear = BilinearLayer(d_model)
        # self.ln1 = nn.RMSNorm(d_model)
        # self.ln2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # TransformerBlock handles residual connections
        x = x + self.dropout(self.attn(x, attention_mask))
        x = x + self.dropout(self.bilinear(x))
        return x


# ============================================================================
# Model Configurations
# ============================================================================


# Assume these are defined elsewhere for a complete, runnable example
@dataclass
class ModelConfig:
    vocab_size: int = None
    d_model: int = 128
    n_ctx: int = 128
    n_head: int = 4
    n_layer: int = 1  # Number of layers
    dropout: float = 0.0
    model_type: str = 'transformer'  # 'transformer', 'attention_only', or 'embed_only'
    use_mixer: bool = False  # Use BilinearMixer instead of QuadraticAttention
    mixer_r: int = None  # Rank/heads for mixer (defaults to n_head if None)
    use_softmax: bool = False  # Use standard softmax attention instead of quadratic
    use_bilinear_attn: bool = False  # Use CausalBilinearSelfAttention (bilinear QK product)
    bilinear_attn_qk_norm: bool = False  # Use RMS norm on QK matrices in CausalBilinearSelfAttention

# --- Cleaned-up and Completed ToyTransformer Class ---

class ToyTransformer(nn.Module):
    """
    A flexible toy transformer supporting different layer configurations.
    This version assumes attention layers handle their own positional encoding (e.g., RoPE).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Build layers based on model type
        layers = []

        if config.model_type == 'embed_only':
            # No layers - just embedding and unembedding
            pass
        elif config.model_type == 'transformer':
            for _ in range(config.n_layer):
                layers.append(TransformerBlock(
                    config.d_model, config.n_head, config.n_ctx, config.dropout,
                    use_mixer=config.use_mixer, mixer_r=config.mixer_r,
                    use_softmax=config.use_softmax, use_bilinear_attn=config.use_bilinear_attn,
                    bilinear_attn_qk_norm=config.bilinear_attn_qk_norm
                ))
        elif config.model_type == 'attention_only':
            for _ in range(config.n_layer):
                if config.use_mixer:
                    r = config.mixer_r if config.mixer_r is not None else config.n_head
                    layers.append(BilinearMixer(seq_len=config.n_ctx, d_model=config.d_model, r=r, causal=True))
                elif config.use_softmax:
                    layers.append(SoftmaxAttention(config.d_model, config.n_head, config.n_ctx))
                elif config.use_bilinear_attn:
                    layers.append(CausalBilinearSelfAttention(config.d_model, config.n_head, config.n_ctx, qk_norm=config.bilinear_attn_qk_norm))
                else:
                    layers.append(QuadraticAttention(config.d_model, config.n_head, config.n_ctx))
        
        self.layers = nn.ModuleList(layers)
        
        # Output head
        # self.ln_f = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # No weight tying - head and embedding are independent

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        """
        Defines the forward pass for the model.

        Args:
            idx (torch.Tensor): Input tensor of shape (batch, sequence_length).
            targets (torch.Tensor, optional): Target tensor for loss calculation.

        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): The model's output logits.
                - loss (torch.Tensor or None): The cross-entropy loss if targets are provided.
        """
        # Token embeddings
        original_x = x
        x = self.embed(x)
        x = self.dropout(x)
        
        # Forward through the layers
        # TransformerBlock handles its own residuals; attention_only layers need residual added here
        for layer in self.layers:
            if isinstance(layer, TransformerBlock):
                x = layer(x, attention_mask)
            elif isinstance(layer, (QuadraticAttention, BilinearMixer, SoftmaxAttention, CausalBilinearSelfAttention)):
                x = x + layer(x, attention_mask)
            else:
                x = x + layer(x) 

        # Output projection
        # x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate loss if targets are provided
        targets = original_x[:, 1:]
        logit_predictions = logits[:, :-1, :]
        # loss = F.cross_entropy(logit_predictions.view(-1, logit_predictions.size(-1)), targets.reshape(-1))    
                # Only calculate loss on non-padded positions if mask is provided
        if attention_mask is not None:
            # Shift mask to align with targets
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
            idx (torch.Tensor): The initial context sequence of shape (batch, sequence_length).
            max_new_tokens (int): The number of new tokens to generate.
            temperature (float): Scaling factor for logits to control randomness.
            top_k (int, optional): If set, only sample from the top k most likely tokens.

        Returns:
            torch.Tensor: The generated sequence including the initial context.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the model's context window
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            
            # Get predictions from the model
            logits, _ = self(idx_cond)
            # Focus only on the logit for the very next token
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set logits for tokens not in the top-k to negative infinity
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the probability distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append the newly sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


# ============================================================================
# Data Loading
# ============================================================================

class StreamingTextDataset:
    """Streaming dataset that tokenizes text on-the-fly"""
    def __init__(self, dataset_name='HuggingFaceFW/fineweb', split='train', 
                 tokenizer_name='gpt2', seq_length=1024, subset='sample-10BT',
                 validation_ratio=0.001, seed=42):
        from transformers import AutoTokenizer
        from datasets import load_dataset
        import hashlib
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_length = seq_length
        self.validation_ratio = validation_ratio
        self.is_validation = (split == 'validation')

        # Load streaming dataset
        # For FineWeb, we only have 'train' split available
        # For SimpleStories, use 'test' for validation
        if 'fineweb' in dataset_name.lower():
            actual_split = 'train'
        elif 'simplestories' in dataset_name.lower() and split == 'validation':
            actual_split = 'test'
        else:
            actual_split = split
        
        if subset:
            self.dataset = load_dataset(dataset_name, subset, split=actual_split, streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, split=actual_split, streaming=True)
        
        # For datasets with only train split, we create our own train/val split
        # using a hash-based deterministic split
        if 'fineweb' in dataset_name.lower() or actual_split == 'train':
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)
            
            # Filter to create train/validation split based on hash of content
            def should_include(example):
                # Use hash of text to deterministically split data
                text = example.get('text', example.get('content', ''))
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                is_val_sample = (hash_val % int(1/validation_ratio)) == 0
                
                # Include if: (we want validation and this is validation) or 
                #            (we want train and this is not validation)
                return is_val_sample == self.is_validation
            
            self.dataset = self.dataset.filter(should_include)
            
        # Create iterator
        self.iterator = iter(self.dataset)
        self.token_buffer = []
        
    def get_batch(self, batch_size, device='cuda'):
        """Get a batch of tokenized sequences"""
        batch_tokens = []
        
        while len(batch_tokens) < batch_size:
            # Refill token buffer if needed
            while len(self.token_buffer) < self.seq_length + 1:
                try:
                    # Get next text sample
                    sample = next(self.iterator)
                    text = sample.get('text', sample.get('content', sample.get('story', '')))

                    # Check if we got empty text - this indicates a key mismatch
                    if not text:
                        print(f"\n=== DATASET KEY ERROR ===")
                        print(f"Could not find text field in sample!")
                        print(f"Available keys in sample: {list(sample.keys())}")
                        print(f"Sample content: {sample}")
                        import sys
                        sys.exit(1)

                    # Tokenize and add to buffer
                    tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=False)
                    self.token_buffer.extend(tokens)
                except StopIteration:
                    # Restart dataset if we run out
                    self.iterator = iter(self.dataset)
                    if len(self.token_buffer) == 0:  # Prevent infinite loop
                        # Add some padding tokens if completely empty
                        self.token_buffer = [self.tokenizer.eos_token_id] * (self.seq_length + 1)
                        break
                except Exception as e:
                    print(f"\n=== UNEXPECTED ERROR IN get_batch ===")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {e}")
                    print(f"Sample keys: {list(sample.keys()) if 'sample' in locals() else 'sample not defined'}")
                    print(f"Sample: {sample if 'sample' in locals() else 'sample not defined'}")
                    import sys
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)
            
            # Extract sequence from buffer
            if len(self.token_buffer) >= self.seq_length + 1:
                seq = self.token_buffer[:self.seq_length + 1]
                batch_tokens.append(seq)
                # Remove processed tokens (with some overlap to maintain context)
                self.token_buffer = self.token_buffer[self.seq_length:]
        
        # Convert to tensors
        batch = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        x = batch[:, :-1]  # Input sequences
        y = batch[:, 1:]   # Target sequences (shifted by 1)
        
        return x, y


# ============================================================================
# Training Script
# ============================================================================

class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Setup optimizer (Muon)
        self.optimizer = Muon(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum
        )
        self.scaler = GradScaler()
        
        # Learning rate schedule with warmup and cooldown
        self.iteration = 0
        self.total_iterations = None
        
    def get_lr(self):
        """Linear warmdown learning rate schedule: constant for first half, linear decay for second half"""
        if self.total_iterations is None:
            # Fallback if total_iterations not set
            return self.config.learning_rate

        # Constant LR for first 50% of training
        warmdown_start = self.total_iterations // 2
        if self.iteration < warmdown_start:
            return self.config.learning_rate

        # Linear decay for the remaining 50%
        decay_progress = (self.iteration - warmdown_start) / (self.total_iterations - warmdown_start)
        return self.config.learning_rate - decay_progress * (self.config.learning_rate - self.config.min_lr)
    
    def train_step(self, input_ids, attention_mask):
        """Single training step"""
        # Set learning rate
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.zero_grad()
        # Forward pass
        with autocast(device_type="cuda"):
            _, loss = self.model(input_ids, attention_mask)

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Unscale gradients for clipping and norm computation
        self.scaler.unscale_(self.optimizer)

        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.iteration += 1

        return loss.item(), lr, grad_norm.item()
    
    def evaluate(self, dataloader, max_batches=50):
        """Evaluate model on validation set"""
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if i >= max_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                losses.append(loss.item())
        self.model.train()
        return np.mean(losses)


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class CausalBilinearSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_head: int, n_ctx: int, qk_norm: bool = False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head
        self.qk_norm = qk_norm
        assert d_model % n_head == 0
        self.c_q = CastedLinear(d_model, d_model, bias=False)
        self.c_k = CastedLinear(d_model, d_model, bias=False)
        self.c_q2 = CastedLinear(d_model, d_model, bias=False)
        self.c_k2 = CastedLinear(d_model, d_model, bias=False)
        self.c_v = CastedLinear(d_model, d_model, bias=False)
        # output projection
        self.c_proj = CastedLinear(d_model, d_model, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        q2 = self.c_q2(x).view(B, T, self.n_head, self.head_dim)
        k2 = self.c_k2(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        if self.qk_norm:
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
            q2, k2 = F.rms_norm(q2, (q2.size(-1),)), F.rms_norm(k2, (k2.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q2, k2 = apply_rotary_emb(q2, cos, sin), apply_rotary_emb(k2, cos, sin)
        y = self.squared_attention(q, k, v, q2, k2)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

    def squared_attention(self, q, k, v, q2, k2):
        B, T, H, D = q.shape # B: batch size, T: sequence length, H: number of heads, D: dimension of each head
        # 64, 1024, 6, 64
        scores = einsum(q, k, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        scores2 = einsum(q2, k2, "... seq_q2 n_head d_head2, ... seq_k2 n_head d_head2 -> ... n_head seq_q2 seq_k2")
        pattern = (scores / D) * (scores2 / D)
        causal_mask = torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool))
        pattern.masked_fill_(causal_mask.logical_not(), 0.0)

        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... n_head seq_q d_head")
        return z
    

@dataclass
class TrainingConfig:
    # Model
    model_config: ModelConfig = None
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.01  # Starting LR
    momentum: float = 0.95
    min_lr: float = 3e-4  # Final LR after linear decay
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    max_epochs: int = 1
    grad_clip: float = 1.0
    
    # Logging
    eval_interval: int = 100
    log_interval: int = 10


def visualize_prediction_error(model, tokenizer, text, pre_tokenized=False, device='cpu'):
    """
    Generates an HTML visualization of prediction loss with a clean white-to-red gradient
    and labeled legend bar from 0 to 7 (with saturation at 7).

    Args:
        model: The pre-trained transformer model.
        tokenizer: The tokenizer corresponding to the model.
        text (str): The input text to analyze.
        device (str): The device to run the model on ('cpu' or 'cuda').
    """
    # 1. Setup and Tokenization
    model.to(device)
    model.eval()
    
    if pre_tokenized:
        tokens = torch.tensor(text, dtype=torch.long, device=device).unsqueeze(0)
    else:
        tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # 2. Model Inference
    with torch.no_grad():
        logits, _ = model(tokens)

    # 3. Calculate Per-Token Loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tokens[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    loss = torch.cat([torch.tensor([0.0], device=device), loss])
    
    # 4. Set up color mapping (white to red, saturating at 7)
    vmin = 0.0
    vmax = 7.0
    
    # Create a custom colormap from white to red
    cmap_colors = ['#ffffff', '#ff0000']  # White to red
    n_bins = 100
    cmap = colors.LinearSegmentedColormap.from_list('white_red', cmap_colors, N=n_bins)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)  # clip=True ensures saturation

    # 5. Generate HTML for each token
    html_spans = []
    # decoded_tokens = [tokenizer.decode(t) for t in tokens[0]]
    def get_token_strings_with_spaces(tokenizer, token_ids):
        """Extract individual token strings with proper spacing preserved"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        decoded_tokens = []
        for i in range(len(token_ids)):
            # Decode up to and including current token
            full_text = tokenizer.decode(token_ids[:i+1])
            
            if i == 0:
                decoded_tokens.append(full_text)
            else:
                # Decode up to but not including current token
                prev_text = tokenizer.decode(token_ids[:i])
                # The difference is the current token's contribution (including any leading space)
                decoded_tokens.append(full_text[len(prev_text):])
        
        return decoded_tokens

    decoded_tokens = get_token_strings_with_spaces(tokenizer, tokens[0])
    print(tokens)
    print(decoded_tokens)
    for i, token_str in enumerate(decoded_tokens):
        token_loss = loss[i].item()
        # Clip the loss value to our range
        clipped_loss = min(token_loss, vmax)
        
        rgba_color = cmap(norm(clipped_loss))
        hex_color = colors.to_hex(rgba_color)

        # Determine text color based on background luminance
        luminance = 0.299*rgba_color[0] + 0.587*rgba_color[1] + 0.114*rgba_color[2]
        text_color = 'white' if luminance < 0.5 else 'black'
        
        # Clean up token display
        display_token = token_str
        
#         # Handle common tokenizer artifacts
#         display_token = display_token.replace('Ġ', ' ')  # GPT-2 style space
#         display_token = display_token.replace('▁', ' ')  # Sentencepiece style space
#         display_token = display_token.replace('Ċ', '\n')  # Newline representation
#         display_token = display_token.replace('ĉ', '\t')  # Tab representation
        
#         # Handle special quote characters that tokenizers sometimes produce
#         display_token = display_token.replace('"', '"')
#         display_token = display_token.replace('"', '"')
#         display_token = display_token.replace(''', "'")
#         display_token = display_token.replace(''', "'")
#         display_token = display_token.replace('′', "'")
#         display_token = display_token.replace('´', "'")
#         display_token = display_token.replace('`', "'")
#         display_token = display_token.replace('â€™', "'")  # Mangled UTF-8 for apostrophe
#         display_token = display_token.replace('â€œ', '"')  # Mangled UTF-8 for left quote
#         display_token = display_token.replace('â€', '"')   # Mangled UTF-8 for right quote
        
        
#         # HTML escape after replacements
#         escaped_token = display_token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        escaped_token = display_token
        
#         # Handle spaces and newlines for display
#         if escaped_token == ' ':
#             escaped_token = '&nbsp;'
#         elif escaped_token == '\n':
#             escaped_token = '↵'  # Use a return symbol for newlines
#         elif escaped_token == '':
#             escaped_token = '∅'  # Use empty set symbol for empty tokens
        
        # Preserve leading/trailing spaces
        escaped_token = escaped_token.replace(' ', '&nbsp;')
        
        style = (
            f"background-color: {hex_color}; color: {text_color}; "
            "padding: 0px 0px; border-radius: 0px; margin: 3px 0px; display: inline-block;"
        )
        html_spans.append(f'<span style="{style}">{escaped_token}</span>')

    # 6. Generate the Clean Legend Bar with Labels
    # Create labels every 0.5 from 0 to 7
    label_values = [i * 0.5 for i in range(15)]  # 0, 0.5, 1.0, ..., 7.0
    
    # Create the gradient bar as a single element
    gradient_id = "loss-gradient"
    
    # Create label divs
    labels_html = ""
    for value in label_values:
        # Calculate position as percentage
        percent_pos = (value / vmax) * 100
        
        # Create tick mark and label
        labels_html += f"""
        <div style="position: absolute; left: {percent_pos}%; transform: translateX(-50%);">
            <div style="width: 1px; height: 8px; background-color: #fff; margin: 0 auto;"></div>
            <div style="margin-top: 2px; font-size: 11px; color: #fff; white-space: nowrap;">
                {value:.1f}
            </div>
        </div>
        """

    # Create the gradient CSS
    gradient_css = f"""
    <style>
        #{gradient_id} {{
            background: linear-gradient(to right, #ffffff 0%, #ff0000 100%);
            height: 25px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
    </style>
    """

    legend_html = f"""
    {gradient_css}
    <div style="font-family: Arial, sans-serif; margin-top: 25px; padding: 20px 0;">
        <p style="margin-bottom: 15px;"><b>Cross-Entropy Loss</b> (0-7 scale, saturates at 7)</p>
        <div style="position: relative; width: 100%; max-width: 600px;">
            <div id="{gradient_id}"></div>
            <div style="position: relative; height: 30px; margin-top: 0;">
                {labels_html}
            </div>
        </div>
    </div>
    """

    # 7. Display the final HTML
    final_html = f"""
    <div style="font-family: Arial, sans-serif;">
        <div style="line-height: 2.0;">{"".join(html_spans)}</div>
        {legend_html}
    </div>
    """
    display(HTML(final_html))
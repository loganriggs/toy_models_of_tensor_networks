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


# ============================================================================
# Muon Optimizer from modded-nanogpt
# ============================================================================

@torch.compile
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
        
    def forward(self, x):
        q, k, v = [rearrange(op(x), '... (n_head d_head) -> ... n_head d_head', 
                            n_head=self.n_head) for op in [self.q, self.k, self.v]]
        
        # Apply rotary embeddings and normalization
        q, k = self.rotary(self.norm(q)), self.rotary(self.norm(k))
        
        # Quadratic scoring function
        scores = einsum(q, k, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        pattern = self.mask((scores / self.d_head).square())
        
        # Aggregate values
        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... seq_q n_head d_head")
        z = rearrange(z, '... seq n_head d_head -> ... seq (n_head d_head)')
        
        return x + self.o(z) * self.scale


class ReLUSquared(nn.Module):
    """ReLU² activation from modded-nanogpt"""
    def forward(self, x):
        return F.relu(x).square()


class BilinearLayer(nn.Module):
    """Bilinear layer: element-wise product of two linear projections
    Similar to SwiGLU but without the gating nonlinearity"""
    def __init__(self, d_model, d_hidden=None, bias=True):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model
        
        # Two parallel projections to hidden dimension
        self.proj1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.proj2 = nn.Linear(d_model, d_hidden, bias=bias)
        
        # Output projection
        self.down = nn.Linear(d_hidden, d_model, bias=bias)
        
        # Zero-initialize down projection for stability
        init.zeros_(self.down.weight)
        if bias:
            init.zeros_(self.down.bias)
            
    def forward(self, x):
        # Compute two parallel projections and multiply element-wise
        hidden = self.proj1(x) * self.proj2(x)
        return self.down(hidden)


class TransformerBlock(nn.Module):
    """Single transformer block with quadratic attention and bilinear layer"""
    def __init__(self, d_model, n_head, n_ctx, dropout=0.0):
        super().__init__()
        self.attn = QuadraticAttention(d_model, n_head, n_ctx)
        self.bilinear = BilinearLayer(d_model)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.bilinear(self.ln2(x)))
        return x


# ============================================================================
# Model Configurations
# ============================================================================

@dataclass
class ModelConfig:
    model_type: Literal['attention_only_1L', 'attention_only_2L', 'transformer_1L', 'transformer_2L']
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    d_model: int = 768
    n_head: int = 12
    n_ctx: int = 1024
    dropout: float = 0.0
    
    def __post_init__(self):
        assert self.d_model % self.n_head == 0


class ToyTransformer(nn.Module):
    """Flexible toy transformer supporting different configurations"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.n_ctx, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Build layers based on model type
        layers = []
        if config.model_type == 'attention_only_1L':
            layers.append(QuadraticAttention(config.d_model, config.n_head, config.n_ctx))
            layers.append(nn.RMSNorm(config.d_model))
        elif config.model_type == 'attention_only_2L':
            for _ in range(2):
                layers.append(QuadraticAttention(config.d_model, config.n_head, config.n_ctx))
                layers.append(nn.RMSNorm(config.d_model))
        elif config.model_type == 'transformer_1L':
            layers.append(TransformerBlock(config.d_model, config.n_head, config.n_ctx, config.dropout))
        elif config.model_type == 'transformer_2L':
            for _ in range(2):
                layers.append(TransformerBlock(config.d_model, config.n_head, config.n_ctx, config.dropout))
        
        self.layers = nn.ModuleList(layers)
        
        # Output head
        self.ln_f = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Token + position embeddings
        x = self.embed(idx) + self.pos_embed(pos)
        x = self.dropout(x)
        
        # Forward through layers
        for layer in self.layers:
            if isinstance(layer, (QuadraticAttention, TransformerBlock)):
                x = layer(x)
            else:  # RMSNorm layers for attention-only models
                x = layer(x)
        
        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss


# ============================================================================
# Data Loading
# ============================================================================

class StreamingTextDataset:
    """Streaming dataset that tokenizes text on-the-fly"""
    def __init__(self, dataset_name='HuggingFaceFW/fineweb', split='train', 
                 tokenizer_name='gpt2', seq_length=1024, subset='sample-10BT'):
        from transformers import AutoTokenizer
        from datasets import load_dataset
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_length = seq_length
        
        # Load streaming dataset
        # Options: 'HuggingFaceFW/fineweb' (subset='sample-10BT' or 'sample-100BT')
        #          'openwebtext' (no subset needed)
        #          'EleutherAI/pile' (subset='all' or specific subset)
        if subset:
            self.dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
            
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
                    text = sample.get('text', sample.get('content', ''))
                    
                    # Tokenize and add to buffer
                    tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=False)
                    self.token_buffer.extend(tokens)
                except StopIteration:
                    # Restart dataset if we run out
                    self.iterator = iter(self.dataset)
            
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
        
        # Learning rate schedule with warmup and cooldown
        self.iteration = 0
        
    def get_lr(self):
        """Cosine learning rate schedule with warmup"""
        if self.iteration < self.config.warmup_iters:
            return self.config.learning_rate * self.iteration / self.config.warmup_iters
        if self.iteration > self.config.lr_decay_iters:
            return self.config.min_lr
        decay_ratio = (self.iteration - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
    
    def train_step(self, x, y):
        """Single training step"""
        # Set learning rate
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        # Forward pass
        logits, loss = self.model(x, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
        # Optimizer step
        self.optimizer.step()
        
        self.iteration += 1
        
        return loss.item(), lr
    
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


@dataclass
class TrainingConfig:
    # Model
    model_config: ModelConfig = None
    
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-3
    momentum: float = 0.95
    min_lr: float = 3e-4  # Don't decay to 0, following speedrun
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    max_iters: int = 5000
    grad_clip: float = 1.0
    
    # Logging
    eval_interval: int = 100
    log_interval: int = 10


def main():
    # Configure model - using real tokenizer vocab size now
    model_config = ModelConfig(
        model_type='transformer_1L',  # Change this to experiment with different architectures
        vocab_size=50257,  # GPT-2 tokenizer size
        d_model=512,  # Moderate size for experiments
        n_head=8,
        n_ctx=512,  # Shorter context for faster training
        dropout=0.1
    )
    
    training_config = TrainingConfig(
        model_config=model_config,
        batch_size=16,  # Adjust based on GPU memory
        learning_rate=3e-3,
        max_iters=10000,
        eval_interval=500,
        log_interval=50
    )
    
    # Create model
    model = ToyTransformer(model_config)
    print(f"Model type: {model_config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    print("Initializing datasets...")
    train_dataset = StreamingTextDataset(
        dataset_name='HuggingFaceFW/fineweb',  # Or 'openwebtext' or 'EleutherAI/pile'
        subset='sample-10BT',  # 10B token sample
        split='train',
        seq_length=model_config.n_ctx
    )
    
    val_dataset = StreamingTextDataset(
        dataset_name='HuggingFaceFW/fineweb',
        subset='sample-10BT',
        split='validation',
        seq_length=model_config.n_ctx
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Training loop
    print("Starting training...")
    for iter in range(training_config.max_iters):
        # Get batch of real data
        x, y = train_dataset.get_batch(training_config.batch_size)
        
        # Train step
        loss, lr = trainer.train_step(x, y)
        
        # Logging
        if iter % training_config.log_interval == 0:
            print(f"Iter {iter}: loss={loss:.4f}, lr={lr:.6f}")
            
        # Evaluation
        if iter % training_config.eval_interval == 0 and iter > 0:
            val_losses = []
            for _ in range(20):  # Evaluate on 20 batches
                x_val, y_val = val_dataset.get_batch(training_config.batch_size)
                _, val_loss = model(x_val, y_val)
                val_losses.append(val_loss.item())
            val_loss = np.mean(val_losses)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Generate sample text
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device='cuda')
            generated = model.generate(context, max_new_tokens=100, temperature=0.8)
            print(f"Sample generation: {train_dataset.tokenizer.decode(generated[0].tolist())}")
            model.train()
    
    print("Training complete!")
    
    # Save model
    torch.save(model.state_dict(), f'toy_transformer_{model_config.model_type}.pt')
    print(f"Model saved to toy_transformer_{model_config.model_type}.pt")
    

if __name__ == "__main__":
    main()
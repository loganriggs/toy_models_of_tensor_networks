"""Minimal transformer with squared bilinear attention (2 Q's, 2 K's) and optional bilinear MLP."""

from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from einops import einsum


@dataclass
class Config:
    vocab_size: int = 4096
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 256
    seq_len: int = 256
    mlp_hidden: int = 512  # h = 2 * n_embd by default
    # Per-block MLP: tuple of bools, e.g. (True, False) = first block has MLP, second doesn't
    # If None, no blocks have MLP
    block_has_mlp: Optional[tuple] = None
    use_rmsnorm: bool = False
    use_qk_norm: bool = False  # normalize Q,K to unit norm before dot product
    use_final_norm: bool = False  # RMSNorm before lm_head only (no per-layer norms)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class Rotary(nn.Module):
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
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class BilinearAttention(nn.Module):
    """Causal squared bilinear self-attention with two Q/K pairs.

    pattern = (Q1@K1^T / D) * (Q2@K2^T / D), causally masked.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.qk_norm = config.use_qk_norm
        assert self.n_embd % self.n_head == 0

        self.q1 = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k1 = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.q2 = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k2 = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.out = nn.Linear(self.n_embd, self.n_embd, bias=False)
        nn.init.zeros_(self.out.weight)

        self.rotary = Rotary(self.head_dim)

    def forward(self, x, return_max_logit=False):
        B, T, C = x.size()
        q1 = self.q1(x).view(B, T, self.n_head, self.head_dim)
        k1 = self.k1(x).view(B, T, self.n_head, self.head_dim)
        q2 = self.q2(x).view(B, T, self.n_head, self.head_dim)
        k2 = self.k2(x).view(B, T, self.n_head, self.head_dim)
        v = self.v(x).view(B, T, self.n_head, self.head_dim)

        cos, sin = self.rotary(q1)
        q1, k1 = apply_rotary_emb(q1, cos, sin), apply_rotary_emb(k1, cos, sin)
        q2, k2 = apply_rotary_emb(q2, cos, sin), apply_rotary_emb(k2, cos, sin)

        if self.qk_norm:
            q1 = F.normalize(q1, dim=-1)
            k1 = F.normalize(k1, dim=-1)
            q2 = F.normalize(q2, dim=-1)
            k2 = F.normalize(k2, dim=-1)

        D = self.head_dim
        scores1 = einsum(q1, k1, "b sq h d, b sk h d -> b h sq sk")
        scores2 = einsum(q2, k2, "b sq h d, b sk h d -> b h sq sk")

        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        pattern = (scores1 / D) * (scores2 / D)
        pattern = pattern.masked_fill(~causal_mask, 0.0)

        y = einsum(pattern, v, "b h sq sk, b sk h d -> b h sq d")
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(y)

        if return_max_logit:
            # Max absolute value of the bilinear attention pattern (before masking with 0)
            max_logit = pattern.abs().max().item()
            return out, max_logit
        return out


class BilinearMLP(nn.Module):
    """Bilinear MLP: Down @ (Left(x) * Right(x))."""

    def __init__(self, config):
        super().__init__()
        self.left = nn.Linear(config.n_embd, config.mlp_hidden, bias=False)
        self.right = nn.Linear(config.n_embd, config.mlp_hidden, bias=False)
        self.down = nn.Linear(config.mlp_hidden, config.n_embd, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        return self.down(self.left(x) * self.right(x))


class Block(nn.Module):
    def __init__(self, config, has_mlp=False):
        super().__init__()
        self.attn = BilinearAttention(config)
        self.mlp = BilinearMLP(config) if has_mlp else None
        if config.use_rmsnorm:
            self.norm_attn = RMSNorm(config.n_embd)
            self.norm_mlp = RMSNorm(config.n_embd) if has_mlp else None
        self.use_rmsnorm = config.use_rmsnorm

    def forward(self, x, return_norms=False):
        norms = {}
        if self.use_rmsnorm:
            attn_out = self.attn(self.norm_attn(x), return_max_logit=return_norms)
        else:
            attn_out = self.attn(x, return_max_logit=return_norms)

        if return_norms:
            attn_out, max_logit = attn_out
            norms["max_attn_logit"] = max_logit

        x = x + attn_out
        if self.mlp is not None:
            if self.use_rmsnorm:
                x = x + self.mlp(self.norm_mlp(x))
            else:
                x = x + self.mlp(x)

        if return_norms:
            # RMS norm of residual stream (mean over batch×seq)
            rms = x.float().pow(2).mean(-1).sqrt().mean().item()
            norms["residual_rms"] = rms

        if return_norms:
            return x, norms
        return x


class BilinearGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)

        mlp_flags = config.block_has_mlp or (False,) * config.n_layer
        assert len(mlp_flags) == config.n_layer
        self.blocks = nn.ModuleList([
            Block(config, has_mlp=mlp_flags[i]) for i in range(config.n_layer)
        ])

        if config.use_rmsnorm or config.use_final_norm:
            self.final_norm = RMSNorm(config.n_embd)
        self.use_rmsnorm = config.use_rmsnorm
        self.use_final_norm = config.use_final_norm

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        nn.init.zeros_(self.lm_head.weight)

    def forward(self, idx, targets=None, return_norms=False):
        x = self.embed(idx)
        norms_dict = {}
        residual_rms_tensors = []  # for norm penalty (backprop-able)

        if return_norms:
            embed_rms = x.float().pow(2).mean(-1).sqrt().mean().item()
            norms_dict["embed_rms"] = embed_rms

        for i, block in enumerate(self.blocks):
            if return_norms:
                x, block_norms = block(x, return_norms=True)
                for k, v in block_norms.items():
                    norms_dict[f"layer{i}_{k}"] = v
                # Also keep a backprop-able version of residual RMS
                residual_rms_tensors.append(
                    x.float().pow(2).mean(-1).sqrt().mean()
                )
            else:
                x = block(x)

        if self.use_rmsnorm or self.use_final_norm:
            x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if return_norms:
            # Mean of squared RMS norms across layers (for norm penalty)
            if residual_rms_tensors:
                norm_loss = sum(r.pow(2) for r in residual_rms_tensors) / len(residual_rms_tensors)
            else:
                norm_loss = torch.tensor(0.0, device=x.device)
            norms_dict["_norm_loss"] = norm_loss  # tensor, not float
            norms_dict["_residual_rms_tensors"] = residual_rms_tensors  # list of tensors
            return logits, loss, norms_dict
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------- Muon optimizer ----------

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    Single-GPU version (no dist.all_reduce).
    Use for 2D weight matrices only. Embed/lm_head/scalars should use AdamW.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = dict(
                svd=zeropower_via_svd,
                newtonschulz5=zeropower_via_newtonschulz5,
            )[group['backend']]

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_backend(g, steps=group['backend_steps'])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr)

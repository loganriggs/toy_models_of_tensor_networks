import torch
import torch.nn as nn


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
        nn.init.normal_(self.out_proj.weight, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (B, S, D) where S <= seq_len
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


# Quick test
if __name__ == "__main__":
    B, S, D, r = 2, 3, 8, 4
    
    mixer = BilinearMixer(seq_len=S, d_model=D, r=r, causal=True)
    x = torch.randn(B, S, D)
    out = mixer(x)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    
    # Verify causality: changing future shouldn't affect past
    x2 = x.clone()
    x2[:, 2, :] = torch.randn(D)  # Change position 2
    out2 = mixer(x2)
    
    print(f"\nCausality check (should be True):")
    print(f"  Position 0 unchanged: {torch.allclose(out[:, 0], out2[:, 0])}")
    print(f"  Position 1 unchanged: {torch.allclose(out[:, 1], out2[:, 1])}")
    print(f"  Position 2 changed:   {not torch.allclose(out[:, 2], out2[:, 2])}")

# Input:  torch.Size([2, 3, 8])
# Output: torch.Size([2, 3, 8])

# Causality check (should be True):
#   Position 0 unchanged: True
#   Position 1 unchanged: True
#   Position 2 changed:   True
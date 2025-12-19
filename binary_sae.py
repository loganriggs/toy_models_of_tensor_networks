"""
Sparse OR-Autoencoder with TopK-Min Sparsity

Minimal implementation for binary matrix factorization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseORAE(nn.Module):
    """
    Sparse OR-Autoencoder for binary data.
    
    Args:
        input_dim: Size of input features (n)
        latent_dim: Number of latent patterns (k)
        top_k: Max features to select per sample
        min_threshold: Only keep features above this threshold
    
    Example:
        model = SparseORAE(input_dim=50, latent_dim=16, top_k=3)
        model.fit(X, epochs=500)
        
        # Get reconstructions
        x_hat = model.reconstruct(X)
        
        # See which features activated
        features = model.get_active_features(X[0:5])
    """
    
    def __init__(
        self, 
        input_dim: int,
        latent_dim: int,
        top_k: int = 3,
        min_threshold: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.top_k = top_k
        self.min_threshold = min_threshold
        
        # Encoder
        self.encoder = nn.Linear(input_dim, latent_dim)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.constant_(self.encoder.bias, 0.5)
        
        # Dictionary (k patterns of size n)
        self.dictionary = nn.Parameter(torch.rand(latent_dim, input_dim) * 0.4 + 0.3)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent representation."""
        z = torch.sigmoid(self.encoder(x))
        return self._apply_topk_min(z)
    
    def _apply_topk_min(self, z: torch.Tensor) -> torch.Tensor:
        """Select top-k features, but only keep those above min_threshold."""
        k = min(self.top_k, self.latent_dim)
        _, indices = torch.topk(z, k, dim=-1)
        
        mask = torch.zeros_like(z)
        mask.scatter_(-1, indices, 1.0)
        mask = mask * (z > self.min_threshold).float()
        
        # Straight-through gradient
        return z * mask - (z * mask).detach() + (z * mask).detach()
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode via soft OR."""
        D = torch.clamp(self.dictionary, 0, 1)
        zD = z.unsqueeze(-1) * D.unsqueeze(0)
        zD = torch.clamp(zD, 0, 0.9999)
        x_hat = 1 - torch.exp(torch.log(1 - zD + 1e-8).sum(dim=-2))
        return torch.clamp(x_hat, 1e-7, 1 - 1e-7)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode -> decode."""
        z = self.encode(x)
        return self.decode(z)
    
    # ==== User-friendly methods ====
    
    def reconstruct(self, x: torch.Tensor, binary: bool = True) -> torch.Tensor:
        """
        Reconstruct input.
        
        Args:
            x: Input tensor (batch, input_dim) or (input_dim,)
            binary: If True, return binary (thresholded at 0.5)
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(x)
            if binary:
                x_hat = (x_hat > 0.5).float()
        
        return x_hat.squeeze(0) if single else x_hat
    
    def get_active_features(self, x: torch.Tensor) -> list[list[tuple[int, float]]]:
        """
        Get active features for each input.
        
        Args:
            x: Input tensor (batch, input_dim) or (input_dim,)
            
        Returns:
            List of lists, where each inner list contains (feature_idx, activation) tuples
            sorted by activation strength.
        
        Example:
            >>> features = model.get_active_features(X[0:3])
            >>> print(features[0])  # First sample
            [(2, 0.89), (7, 0.72)]  # Feature 2 active at 0.89, feature 7 at 0.72
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
        
        results = []
        for i in range(z.shape[0]):
            active = []
            for j in range(z.shape[1]):
                if z[i, j] > 0.01:
                    active.append((j, z[i, j].item()))
            active.sort(key=lambda x: -x[1])  # Sort by activation descending
            results.append(active)
        
        return results[0] if single else results
    
    def get_pattern(self, feature_idx: int, binary: bool = True) -> torch.Tensor:
        """
        Get dictionary pattern for a specific feature.
        
        Args:
            feature_idx: Index of the latent feature
            binary: If True, binarize at 0.5
        """
        with torch.no_grad():
            pattern = torch.clamp(self.dictionary[feature_idx], 0, 1)
            if binary:
                pattern = (pattern > 0.5).float()
            return pattern
    
    def get_all_patterns(self, binary: bool = True) -> torch.Tensor:
        """Get all dictionary patterns (latent_dim, input_dim)."""
        with torch.no_grad():
            D = torch.clamp(self.dictionary, 0, 1)
            if binary:
                D = (D > 0.5).float()
            return D
    
    def explain(self, x: torch.Tensor) -> dict:
        """
        Explain reconstruction for a single input.
        
        Returns dict with:
            - input: Original input
            - reconstruction: Reconstructed output
            - active_features: List of (idx, activation) tuples
            - patterns: Dict mapping feature_idx -> pattern
        """
        assert x.dim() == 1, "Pass a single sample"
        
        self.eval()
        with torch.no_grad():
            x_hat = self.reconstruct(x, binary=True)
            active = self.get_active_features(x)
            patterns = {idx: self.get_pattern(idx) for idx, _ in active}
        
        return {
            'input': x,
            'reconstruction': x_hat,
            'active_features': active,
            'patterns': patterns,
            'accuracy': (x_hat == x).float().mean().item(),
        }
    
    # ==== Training ====
    
    def fit(
        self,
        X: torch.Tensor,
        epochs: int = 500,
        batch_size: int = 128,
        lr: float = 0.01,
        l1_weight: float = 0.001,
        verbose: bool = True,
        print_every: int = 100,
    ) -> dict:
        """
        Train the model.
        
        Args:
            X: Binary input matrix (n_samples, input_dim)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            l1_weight: L1 regularization weight
            verbose: Print progress
            print_every: Print frequency
            
        Returns:
            Training history dict
        """
        device = next(self.parameters()).device
        X = X.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_samples = X.shape[0]
        
        history = {'loss': [], 'accuracy': []}
        best_acc, best_state = 0.0, None
        
        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0
            
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                x_batch = X[idx]
                
                optimizer.zero_grad()
                
                z = self.encode(x_batch)
                x_hat = self.decode(z)
                
                loss = F.binary_cross_entropy(x_hat, x_batch)
                loss = loss + l1_weight * torch.sigmoid(self.encoder(x_batch)).mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                with torch.no_grad():
                    self.dictionary.data.clamp_(0, 1)
                
                epoch_loss += loss.item()
            
            # Evaluate
            self.eval()
            with torch.no_grad():
                x_hat = self.forward(X)
                acc = ((x_hat > 0.5).float() == X).float().mean().item()
                avg_loss = epoch_loss / (n_samples / batch_size)
                
                history['loss'].append(avg_loss)
                history['accuracy'].append(acc)
                
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
            
            if verbose and (epoch + 1) % print_every == 0:
                z = self.encode(X)
                active = (z > 0.01).sum(dim=1).float()
                print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | "
                      f"Acc: {acc:.4f} | Active: {active.mean():.1f} [{active.min():.0f}-{active.max():.0f}]")
        
        if best_state:
            self.load_state_dict(best_state)
        
        if verbose:
            print(f"\nBest accuracy: {best_acc:.4f}")
        
        return history


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    
    # Generate synthetic OR data
    torch.manual_seed(42)
    n_samples, n_features, n_patterns = 5000, 50, 8
    
    D_true = (torch.rand(n_patterns, n_features) < 0.15).float()
    Z_true = (torch.rand(n_samples, n_patterns) < 0.12).float()
    for i in range(n_samples):
        if Z_true[i].sum() == 0:
            Z_true[i, torch.randint(n_patterns, (1,))] = 1
    
    X = torch.zeros(n_samples, n_features)
    for i in range(n_samples):
        for k in torch.where(Z_true[i] > 0)[0]:
            X[i] = torch.maximum(X[i], D_true[k])
    
    print(f"Data: {n_samples} samples, {n_features} features, {n_patterns} true patterns")
    print(f"True active per sample: {Z_true.sum(dim=1).mean():.2f}")
    
    # Create and train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SparseORAE(
        input_dim=n_features,
        latent_dim=16,
        top_k=3,
        min_threshold=0.1,
    ).to(device)
    
    model.fit(X, epochs=500, print_every=100)
    
    # Demo: explain a single sample
    print("\n" + "="*60)
    print("EXAMPLE: Explaining sample 0")
    print("="*60)
    
    result = model.explain(X[0].to(device))
    
    print(f"\nInput:   {X[0].numpy().astype(int)}")
    print(f"Recon:   {result['reconstruction'].cpu().numpy().astype(int)}")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"\nActive features: {result['active_features']}")
    
    for idx, activation in result['active_features']:
        pattern = result['patterns'][idx].cpu().numpy().astype(int)
        print(f"  Feature {idx} (act={activation:.3f}): {pattern}")
    
    # Demo: batch query
    print("\n" + "="*60)
    print("EXAMPLE: Active features for samples 0-4")
    print("="*60)
    
    features = model.get_active_features(X[0:5].to(device))
    for i, f in enumerate(features):
        print(f"Sample {i}: {f}")
"""Tucker decomposition pipeline for bilinear layer interpretability.

Stages 1-2: Synthetic CP tensor verification + AND-gate task.
Run: python tucker/tucker_pipeline.py
"""

# %% Imports and setup

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly
from tensorly.decomposition import tucker as tensorly_tucker
from tensorly.decomposition import parafac

tensorly.set_backend("pytorch")

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)


# %% Core Tucker pipeline

def hosvd(W, ranks):
    """Higher-Order SVD via mode-wise SVD truncation.

    This is the true HOSVD (not ALS). For each mode, compute SVD of the
    mode-k unfolding and keep the top r_k left singular vectors as the
    factor matrix. The core is obtained by projecting W onto all factors.

    Args:
        W: tensor of shape (d1, d2, d3)
        ranks: tuple (r1, r2, r3) or int applied to all modes

    Returns:
        G: core tensor of shape (r1, r2, r3)
        factors: list of [A, B, C] factor matrices
    """
    if isinstance(ranks, int):
        ranks = (ranks, ranks, ranks)

    factors = []
    for mode in range(3):
        # Mode-k unfolding
        unfolded = tensorly.unfold(W, mode)
        U, S, Vh = torch.linalg.svd(unfolded, full_matrices=False)
        factors.append(U[:, :ranks[mode]])

    A, B, C = factors
    # Core G = W x_0 A^T x_1 B^T x_2 C^T
    G = torch.einsum("ijk,ia,jb,kc->abc", W, A, B, C)

    return G, factors


def rotate_core(G, R_A, R_B, R_C):
    """Apply rotations to core: G_rot = G x_1 R_A^T x_2 R_B^T x_3 R_C^T."""
    return torch.einsum("pqr,ap,bq,cr->abc", G, R_A, R_B, R_C)


def off_diagonal_mass(G):
    """Fraction of Frobenius norm in off-superdiagonal entries."""
    r = min(G.shape)
    diag_vals = torch.tensor([G[i, i, i] for i in range(r)])
    diag_norm_sq = (diag_vals**2).sum()
    total_norm_sq = (G**2).sum()
    if total_norm_sq < 1e-12:
        return 0.0
    return (1.0 - diag_norm_sq / total_norm_sq).item()


def _givens_rotation(n, i, j, theta):
    """Create an n×n Givens rotation matrix rotating in the (i,j) plane."""
    R = torch.eye(n)
    c, s = torch.cos(theta), torch.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def find_sparse_rotation(G, n_restarts=10, n_sweeps=50):
    """Find rotations R_A, R_B, R_C that make G approximately superdiagonal.

    Uses Jacobi sweeps with Givens rotations: for each pair (i,j) in each mode,
    find the angle that maximizes diagonal energy sum_k G_rot[k,k,k]^2.
    This handles sign correctly (maximizes squared diagonal, not trace).
    """
    r_A, r_B, r_C = G.shape
    r_min = min(r_A, r_B, r_C)

    best_odm = float("inf")
    best_rotations = None

    for restart in range(n_restarts):
        if restart == 0:
            R_A = torch.eye(r_A)
            R_B = torch.eye(r_B)
            R_C = torch.eye(r_C)
        else:
            R_A = torch.linalg.qr(torch.randn(r_A, r_A))[0]
            R_B = torch.linalg.qr(torch.randn(r_B, r_B))[0]
            R_C = torch.linalg.qr(torch.randn(r_C, r_C))[0]

        for _ in range(n_sweeps):
            # Sweep over all Givens pairs for each mode
            for mode_R, r_mode in [(0, r_A), (1, r_B), (2, r_C)]:
                for i in range(r_mode):
                    for j in range(i + 1, r_mode):
                        # Line search over angle theta
                        rotations = [R_A, R_B, R_C]
                        best_theta = 0.0
                        best_diag_energy = -1.0

                        for theta in torch.linspace(-torch.pi, torch.pi, 36):
                            Rij = _givens_rotation(r_mode, i, j, theta)
                            test_R = rotations[mode_R] @ Rij
                            test_rotations = list(rotations)
                            test_rotations[mode_R] = test_R
                            G_test = rotate_core(G, *test_rotations)
                            diag_energy = sum(
                                G_test[k, k, k] ** 2 for k in range(r_min)
                            )
                            if diag_energy > best_diag_energy:
                                best_diag_energy = diag_energy
                                best_theta = theta

                        # Apply best rotation
                        if best_theta != 0.0:
                            Rij = _givens_rotation(r_mode, i, j, best_theta)
                            if mode_R == 0:
                                R_A = R_A @ Rij
                            elif mode_R == 1:
                                R_B = R_B @ Rij
                            else:
                                R_C = R_C @ Rij

        G_rot = rotate_core(G, R_A, R_B, R_C)
        odm = off_diagonal_mass(G_rot)

        if odm < best_odm:
            best_odm = odm
            best_rotations = (R_A.clone(), R_B.clone(), R_C.clone())

    return best_rotations


def tucker_pipeline(W, ranks, fast=False):
    """Full Tucker + rotation pipeline.

    Args:
        W: weight tensor (d_in, d_in, d_out)
        ranks: tuple (r_A, r_B, r_C) or int
        fast: if True, use fewer restarts/sweeps (for rank search)

    Returns:
        G_rot, A_rot, B_rot, C_rot, metrics dict
    """
    if isinstance(ranks, int):
        ranks = (ranks, ranks, ranks)

    # Step 1: HOSVD
    G, (A, B, C) = hosvd(W, ranks)

    odm_before = off_diagonal_mass(G)

    # Step 2: Find sparse rotation
    n_restarts = 3 if fast else 10
    n_sweeps = 10 if fast else 50
    R_A, R_B, R_C = find_sparse_rotation(G, n_restarts=n_restarts, n_sweeps=n_sweeps)

    # Step 3: Apply rotations
    # rotate_core does G x_0 R_A x_1 R_B x_2 R_C, so factors get R^T
    G_rot = rotate_core(G, R_A, R_B, R_C)
    A_rot = A @ R_A.T
    B_rot = B @ R_B.T
    C_rot = C @ R_C.T

    odm_after = off_diagonal_mass(G_rot)

    # Reconstruction error
    W_recon = torch.einsum("pqr,ip,jq,kr->ijk", G_rot, A_rot, B_rot, C_rot)
    recon_error = (W - W_recon).norm() / W.norm()

    metrics = {
        "odm_before": odm_before,
        "odm_after": odm_after,
        "recon_error": recon_error.item(),
        "ranks": ranks,
    }

    return G_rot, A_rot, B_rot, C_rot, metrics


# %% Stage 1: Synthetic CP Tensor Verification

def make_cp_tensor(d_in, d_out, rank, orthogonal=True):
    """Create a known CP tensor for testing."""
    if orthogonal:
        A = torch.linalg.qr(torch.randn(d_in, rank))[0]
        B = torch.linalg.qr(torch.randn(d_in, rank))[0]
    else:
        A = F.normalize(torch.randn(d_in, rank), dim=0)
        B = F.normalize(torch.randn(d_in, rank), dim=0)
    C = torch.randn(d_out, rank)
    W = torch.einsum("ir,jr,kr->ijk", A, B, C)
    return W, A, B, C


def run_stage1():
    """Stage 1: Verify Tucker recovery on synthetic CP tensors."""
    print("=" * 60)
    print("STAGE 1: Synthetic CP Tensor Verification")
    print("=" * 60)

    d_in, d_out, rank = 8, 6, 3
    results = {}

    for label, ortho in [("1A-ortho", True), ("1A-nonortho", False)]:
        print(f"\n--- {label} ---")
        W, A_true, B_true, C_true = make_cp_tensor(d_in, d_out, rank, orthogonal=ortho)

        G_rot, A_rot, B_rot, C_rot, metrics = tucker_pipeline(W, rank)

        print(f"  Reconstruction error: {metrics['recon_error']:.6f}")
        print(f"  Off-diag mass before rotation: {metrics['odm_before']:.4f}")
        print(f"  Off-diag mass after rotation:  {metrics['odm_after']:.4f}")

        # Check factor recovery (up to permutation and sign)
        # Compute absolute cosine similarities between recovered and true factors
        cos_A = (A_rot.T @ A_true).abs()  # (rank, rank)
        cos_B = (B_rot.T @ B_true).abs()
        cos_C = (C_rot.T @ C_true).abs()

        # Best matching (greedy)
        best_cos_A = cos_A.max(dim=1).values.mean().item()
        best_cos_B = cos_B.max(dim=1).values.mean().item()
        best_cos_C = cos_C.max(dim=1).values.mean().item()

        print(f"  Factor recovery (mean best cosine sim):")
        print(f"    A: {best_cos_A:.4f}  B: {best_cos_B:.4f}  C: {best_cos_C:.4f}")

        # Core structure
        print(f"  Core diagonal: {[f'{G_rot[i,i,i].item():.3f}' for i in range(rank)]}")

        results[label] = {**metrics, "cos_A": best_cos_A, "cos_B": best_cos_B, "cos_C": best_cos_C}

    return results


# %% Stage 1 - Rank search visualization

def run_stage1_rank_search():
    """Test Tucker at different ranks to find the elbow."""
    print("\n--- Stage 1: Rank Search ---")

    d_in, d_out, true_rank = 8, 6, 3
    W, _, _, _ = make_cp_tensor(d_in, d_out, true_rank, orthogonal=False)

    ranks_to_try = range(1, 7)
    recon_errors = []
    odm_values = []

    for r in ranks_to_try:
        _, _, _, _, metrics = tucker_pipeline(W, r)
        recon_errors.append(metrics["recon_error"])
        odm_values.append(metrics["odm_after"])
        print(f"  rank={r}: recon_err={metrics['recon_error']:.6f}, odm={metrics['odm_after']:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(list(ranks_to_try), recon_errors, "o-")
    ax1.axvline(true_rank, color="r", linestyle="--", label=f"True rank={true_rank}")
    ax1.set_xlabel("Tucker rank")
    ax1.set_ylabel("Relative reconstruction error")
    ax1.set_title("Rank search — reconstruction error")
    ax1.legend()

    ax2.plot(list(ranks_to_try), odm_values, "o-")
    ax2.axvline(true_rank, color="r", linestyle="--", label=f"True rank={true_rank}")
    ax2.set_xlabel("Tucker rank")
    ax2.set_ylabel("Off-diagonal mass")
    ax2.set_title("Rank search — core sparsity")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage1_rank_search.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage1_rank_search.png'}")


# %% Stage 2: AND-gate task

class BilinearLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.L = nn.Linear(d_in, d_hidden, bias=False)
        self.R = nn.Linear(d_in, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.out(self.L(x) * self.R(x))

    def weight_tensor(self):
        """W[i,j,k] = sum_h L[h,i]*R[h,j]*out[k,h]"""
        return torch.einsum("hi,hj,kh->ijk", self.L.weight, self.R.weight, self.out.weight)


def make_and_gate_dataset():
    """5 boolean inputs, 3 AND-gate outputs.
    Output 0 = feature_0 AND feature_1
    Output 1 = feature_2 AND feature_3
    Output 2 = feature_0 AND feature_4
    """
    inputs = torch.tensor(list(itertools.product([0, 1], repeat=5)), dtype=torch.float32)
    targets = torch.stack(
        [
            inputs[:, 0] * inputs[:, 1],
            inputs[:, 2] * inputs[:, 3],
            inputs[:, 0] * inputs[:, 4],
        ],
        dim=1,
    )
    return inputs, targets


def train_and_gate_model(d_hidden=16, n_epochs=5000, lr=1e-3):
    """Train a bilinear layer on the AND-gate task."""
    inputs, targets = make_and_gate_dataset()
    model = BilinearLayer(d_in=5, d_hidden=d_hidden, d_out=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        pred = model(inputs)
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    # Final accuracy check
    with torch.no_grad():
        pred = model(inputs)
        acc = ((pred > 0.5).float() == targets).float().mean().item()
        print(f"  Final accuracy: {acc:.4f}")

    return model


def run_stage2():
    """Stage 2: AND-gate task with single bilinear layer.

    Uses d_hidden=3 (minimal) so the weight tensor is exactly rank 3,
    isolating the Tucker diagonalization challenge from overparameterization noise.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: AND-gate Task — Single Bilinear Layer")
    print("=" * 60)

    print("\nTraining bilinear layer (d_hidden=3, minimal rank)...")
    model = train_and_gate_model(d_hidden=3)

    W = model.weight_tensor().detach()
    print(f"\nWeight tensor shape: {W.shape}")
    print(f"Weight tensor Frobenius norm: {W.norm():.4f}")

    # Rank search
    print("\n--- Rank search ---")
    max_rank = min(W.shape)
    ranks_to_try = range(1, min(max_rank + 1, 8))
    recon_errors = []
    odm_values = []

    for r in ranks_to_try:
        G_rot, A_rot, B_rot, C_rot, metrics = tucker_pipeline(W, r)
        recon_errors.append(metrics["recon_error"])
        odm_values.append(metrics["odm_after"])
        print(f"  rank={r}: recon_err={metrics['recon_error']:.6f}, odm={metrics['odm_after']:.4f}")

    # Plot rank search
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(list(ranks_to_try), recon_errors, "o-")
    ax1.axvline(3, color="r", linestyle="--", label="True rank=3")
    ax1.set_xlabel("Tucker rank")
    ax1.set_ylabel("Relative reconstruction error")
    ax1.set_title("AND-gate: rank search")
    ax1.legend()

    ax2.plot(list(ranks_to_try), odm_values, "o-")
    ax2.axvline(3, color="r", linestyle="--", label="True rank=3")
    ax2.set_xlabel("Tucker rank")
    ax2.set_ylabel("Off-diagonal mass")
    ax2.set_title("AND-gate: core sparsity")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage2_rank_search.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage2_rank_search.png'}")

    # Detailed analysis at rank=3
    print("\n--- Analysis at rank=3 ---")
    G_rot, A_rot, B_rot, C_rot, metrics = tucker_pipeline(W, 3)
    print(f"  Reconstruction error: {metrics['recon_error']:.6f}")
    print(f"  Off-diag mass: {metrics['odm_after']:.4f}")
    print(f"  Core diagonal: {[f'{G_rot[i,i,i].item():.3f}' for i in range(3)]}")

    # Check factor alignment with true features (one-hot vectors)
    feature_labels = ["f0", "f1", "f2", "f3", "f4"]
    print(f"\n  Left factor A_rot (top feature per component):")
    for c in range(3):
        col = A_rot[:, c].abs()
        top_idx = col.argmax().item()
        print(f"    Component {c}: {feature_labels[top_idx]} (|weight|={col[top_idx]:.3f})")

    print(f"\n  Right factor B_rot (top feature per component):")
    for c in range(3):
        col = B_rot[:, c].abs()
        top_idx = col.argmax().item()
        print(f"    Component {c}: {feature_labels[top_idx]} (|weight|={col[top_idx]:.3f})")

    print(f"\n  Output factor C_rot (top output per component):")
    out_labels = ["out0(f0&f1)", "out1(f2&f3)", "out2(f0&f4)"]
    for c in range(3):
        col = C_rot[:, c].abs()
        top_idx = col.argmax().item()
        print(f"    Component {c}: {out_labels[top_idx]} (|weight|={col[top_idx]:.3f})")

    # Shared feature check: outputs 0 and 2 both use feature_0
    # Find which components map to outputs 0 and 2
    out0_comp = C_rot[0, :].abs().argmax().item()
    out2_comp = C_rot[2, :].abs().argmax().item()
    left_cos = F.cosine_similarity(
        A_rot[:, out0_comp].unsqueeze(0), A_rot[:, out2_comp].unsqueeze(0)
    ).item()
    print(f"\n  Shared feature check:")
    print(f"    Component for out0: {out0_comp}, Component for out2: {out2_comp}")
    print(f"    Cosine(A[:,{out0_comp}], A[:,{out2_comp}]) = {left_cos:.4f}")
    print(f"    (Expected: high if both use feature_0 as left input)")

    # --- Visualize raw weight tensor W (ground truth) ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmax_w = W.abs().max().item()
    for k in range(3):
        W_slice = W[:, :, k].detach().numpy()
        im = axes[k].imshow(W_slice, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w)
        axes[k].set_title(f"Output {k}: {out_labels[k]}", fontsize=11)
        axes[k].set_xlabel("Right input feature")
        axes[k].set_ylabel("Left input feature")
        axes[k].set_xticks(range(5))
        axes[k].set_xticklabels(feature_labels)
        axes[k].set_yticks(range(5))
        axes[k].set_yticklabels(feature_labels)
        for i in range(5):
            for j in range(5):
                val = W_slice[i, j]
                if abs(val) > 0.05:
                    color = "white" if abs(val) > vmax_w * 0.6 else "black"
                    axes[k].text(j, i, f"{val:.2f}", ha="center", va="center",
                                color=color, fontsize=8)
        plt.colorbar(im, ax=axes[k], shrink=0.8)
    plt.suptitle("Raw weight tensor W[:,:,k] — each slice is one output\n"
                 "(W[i,j,k] = contribution to output k from left_i × right_j)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage2_weight_tensor.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage2_weight_tensor.png'}")

    # --- Identify which component maps to which AND gate ---
    # Use C_rot (output factor) to assign: each component's strongest output
    gate_names = ["f0 AND f1", "f2 AND f3", "f0 AND f4"]
    comp_to_gate = {}
    for c in range(3):
        top_out = C_rot[:, c].abs().argmax().item()
        comp_to_gate[c] = gate_names[top_out]
    comp_labels = [f"C{c}: {comp_to_gate[c]}" for c in range(3)]
    print(f"\n  Component assignments: {comp_labels}")

    # --- Visualize core tensor with task labels ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    vmax = G_rot.abs().max().item()
    for r_idx in range(3):
        G_slice = G_rot[:, :, r_idx].detach().numpy()
        im = axes[r_idx].imshow(G_slice, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[r_idx].set_title(f"Output slice {r_idx}: {comp_to_gate[r_idx]}", fontsize=10)
        axes[r_idx].set_xlabel("Right input component")
        axes[r_idx].set_ylabel("Left input component")
        axes[r_idx].set_xticks(range(3))
        axes[r_idx].set_xticklabels([f"C{i}" for i in range(3)], fontsize=9)
        axes[r_idx].set_yticks(range(3))
        axes[r_idx].set_yticklabels([f"C{i}" for i in range(3)], fontsize=9)
        # Annotate each cell with its value
        for i in range(3):
            for j in range(3):
                val = G_slice[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                axes[r_idx].text(j, i, f"{val:.2f}", ha="center", va="center",
                                color=color, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=axes[r_idx], shrink=0.8)

    diag_energy = sum(G_rot[i, i, i].item() ** 2 for i in range(3))
    total_energy = (G_rot ** 2).sum().item()
    plt.suptitle(
        f"Core tensor G — diagonal explains {diag_energy/total_energy*100:.0f}% of energy\n"
        f"(ODM={metrics['odm_after']:.2f} — f0 shared between gates 0 and 2 prevents full diagonalization)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage2_core_tensor.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage2_core_tensor.png'}")

    # --- Visualize factors as bar charts per component ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    bar_colors = ["#d62728", "#2ca02c", "#1f77b4"]

    for c in range(3):
        # Left factor (A_rot): which input features does this component select?
        ax = axes[0][c]
        vals = A_rot[:, c].detach().numpy()
        bars = ax.barh(range(5), vals, color=bar_colors[c], alpha=0.8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(feature_labels)
        ax.set_title(f"C{c} left input: {comp_to_gate[c]}", fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(-1, 1)

        # Right factor (B_rot)
        ax = axes[1][c]
        vals = B_rot[:, c].detach().numpy()
        ax.barh(range(5), vals, color=bar_colors[c], alpha=0.8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(feature_labels)
        ax.set_title(f"C{c} right input: {comp_to_gate[c]}", fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(-1, 1)

        # Output factor (C_rot)
        ax = axes[2][c]
        vals = C_rot[:, c].detach().numpy()
        ax.barh(range(3), vals, color=bar_colors[c], alpha=0.8)
        ax.set_yticks(range(3))
        ax.set_yticklabels(out_labels)
        ax.set_title(f"C{c} output: {comp_to_gate[c]}", fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(-1.2, 1.2)

    axes[0][0].set_ylabel("Left factor A", fontsize=11, fontweight="bold")
    axes[1][0].set_ylabel("Right factor B", fontsize=11, fontweight="bold")
    axes[2][0].set_ylabel("Output factor C", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Tucker factors — each column is one circuit (component)\n"
        "Reading: G[i,i,i] * A[:,i] * B[:,i] → C[:,i]",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage2_factors.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage2_factors.png'}")

    return model, G_rot, A_rot, B_rot, C_rot, metrics


# %% Stage 3: Gated Tucker as Architecture

class GatedTuckerBilinear(nn.Module):
    def __init__(self, d_in, d_out, r_A, r_B, r_C, gate_temp=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_in, r_A) * 0.1)
        self.B = nn.Parameter(torch.randn(d_in, r_B) * 0.1)
        self.C = nn.Parameter(torch.randn(d_out, r_C) * 0.1)
        self.G = nn.Parameter(torch.randn(r_A, r_B, r_C) * 0.1)
        self.gate_logits = nn.Parameter(torch.ones(r_A, r_B, r_C) * 3.0)  # start open
        self.temp = gate_temp

    def gates(self, hard=False):
        if hard:
            return (self.gate_logits > 0).float()
        return torch.sigmoid(self.gate_logits / self.temp)

    def forward(self, x, hard_gates=False):
        G_eff = self.G * self.gates(hard=hard_gates)
        left = x @ self.A    # (batch, r_A)
        right = x @ self.B   # (batch, r_B)
        core_out = torch.einsum("pqr,bp,bq->br", G_eff, left, right)  # (batch, r_C)
        out = core_out @ self.C.T  # (batch, d_out)
        return out

    def gate_loss(self):
        return self.gates().sum()

    def n_active_gates(self):
        return (self.gate_logits > 0).sum().item()


def train_gated_tucker(d_in=5, d_out=3, r=5, n_epochs=10000, lr=1e-3, gate_lambda=1e-4):
    """Train Gated Tucker on AND-gate task.

    Three-phase: warmup (learn task), sparsify (add L1 on G*gate), fine-tune.
    Key insight: instead of pushing gate_logits negative, penalize |G * gate|
    so unused core entries shrink to zero, then prune.
    """
    inputs, targets = make_and_gate_dataset()
    model = GatedTuckerBilinear(d_in, d_out, r, r, r, gate_temp=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Phase 1: Learn the task (no sparsity)
    warmup = n_epochs // 3
    model.temp = 1.0
    for epoch in range(warmup):
        pred = model(inputs)
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2000 == 0:
            print(f"  Phase 1 Epoch {epoch+1}: task={loss.item():.6f}, "
                  f"active={model.n_active_gates()}")

    # Phase 2: Add L1 on effective core entries to encourage sparsity
    for epoch in range(warmup, 2 * warmup):
        pred = model(inputs)
        task_loss = F.mse_loss(pred, targets)
        # L1 on |G * gates| encourages unused entries to have small G
        effective_core = model.G.abs() * model.gates()
        sparsity_loss = effective_core.sum()
        loss = task_loss + gate_lambda * sparsity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2000 == 0:
            n_small = (effective_core < 0.01).sum().item()
            print(f"  Phase 2 Epoch {epoch+1}: task={task_loss.item():.6f}, "
                  f"|G*gate|_1={sparsity_loss.item():.2f}, "
                  f"small_entries={n_small}/{effective_core.numel()}")

    # Prune: zero out small core entries by setting their gate logits to -inf
    with torch.no_grad():
        effective = (model.G.abs() * model.gates()).detach()
        threshold = effective.max() * 0.05  # keep entries > 5% of max
        mask = effective > threshold
        model.gate_logits.data[~mask] = -100.0
        n_kept = mask.sum().item()
        print(f"  Pruned: kept {n_kept}/{mask.numel()} entries (threshold={threshold:.4f})")

    # Phase 3: Fine-tune with hard gates
    model.gate_logits.requires_grad_(False)
    optimizer_ft = torch.optim.Adam(
        [p for n, p in model.named_parameters() if "gate" not in n], lr=lr * 0.1
    )
    for epoch in range(n_epochs - 2 * warmup):
        pred = model(inputs, hard_gates=True)
        loss = F.mse_loss(pred, targets)
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()

    with torch.no_grad():
        pred = model(inputs, hard_gates=True)
        acc = ((pred > 0.5).float() == targets).float().mean().item()
        print(f"  Final accuracy (hard gates): {acc:.4f}")
        print(f"  Active gates: {model.n_active_gates()}")

    return model


def run_stage3():
    """Stage 3: Gated Tucker as architecture."""
    print("\n" + "=" * 60)
    print("STAGE 3: Gated Tucker Architecture")
    print("=" * 60)

    print("\nTraining Gated Tucker (r=5, d_hidden >> true rank)...")
    model = train_gated_tucker(r=5, gate_lambda=0.01)

    # Analyze gate pattern
    gates = model.gates(hard=True).detach()
    active_entries = gates.nonzero()
    print(f"\n  Active gate entries ({len(active_entries)}):")
    for entry in active_entries:
        p, q, r = entry.tolist()
        g_val = model.G[p, q, r].item()
        print(f"    G[{p},{q},{r}] = {g_val:.4f}")

    # Factor directions for active entries
    feature_labels = ["f0", "f1", "f2", "f3", "f4"]
    out_labels = ["out0(f0&f1)", "out1(f2&f3)", "out2(f0&f4)"]

    print(f"\n  Recovered circuits:")
    for entry in active_entries:
        p, q, r = entry.tolist()
        left_feat = model.A[:, p].abs().argmax().item()
        right_feat = model.B[:, q].abs().argmax().item()
        out_feat = model.C[:, r].abs().argmax().item()
        print(f"    {feature_labels[left_feat]} x {feature_labels[right_feat]} -> {out_labels[out_feat]}")

    # Stress test with d_hidden=32
    print("\n--- Stress test: r=8 (very overcomplete) ---")
    model_big = train_gated_tucker(r=8, gate_lambda=0.02)
    gates_big = model_big.gates(hard=True).detach()
    print(f"  Active gates: {gates_big.sum().item():.0f} / {gates_big.numel()}")

    # Visualize gate pattern
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, title in [(axes[0], model, "r=5"), (axes[1], model_big, "r=8")]:
        g = m.gates(hard=True).detach()
        # Sum over one dimension for visualization
        g_sum = g.sum(dim=2)  # (r_A, r_B)
        im = ax.imshow(g_sum.numpy(), cmap="Blues", vmin=0)
        ax.set_title(f"Gate pattern ({title}), active={g.sum().item():.0f}")
        ax.set_xlabel("Right component")
        ax.set_ylabel("Left component")
        plt.colorbar(im, ax=ax)

    plt.suptitle("Stage 3: Gated Tucker gate patterns")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage3_gate_patterns.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage3_gate_patterns.png'}")

    return model


# %% Stage 4: Two-Layer Stacked Bilinear

class TwoLayerBilinear(nn.Module):
    def __init__(self, d_in, d_hidden, d_mid, d_out):
        super().__init__()
        self.layer1 = BilinearLayer(d_in, d_hidden, d_mid)
        self.layer2 = BilinearLayer(d_mid, d_hidden, d_out)

    def forward(self, x):
        r1 = x[:, :self.layer2.L.in_features]  # residual (may need padding)
        h = self.layer1(x)
        r1 = h + x[:, :h.shape[1]] if x.shape[1] >= h.shape[1] else h
        z = self.layer2(r1)
        return z


class TwoLayerBilinearResidual(nn.Module):
    """Two-layer bilinear with proper residual stream."""
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        # d_in = d_hidden_mid = d_out for clean residual
        self.L1 = nn.Linear(d_in, d_hidden, bias=False)
        self.R1 = nn.Linear(d_in, d_hidden, bias=False)
        self.out1 = nn.Linear(d_hidden, d_in, bias=False)  # project back to residual dim

        self.L2 = nn.Linear(d_in, d_hidden, bias=False)
        self.R2 = nn.Linear(d_in, d_hidden, bias=False)
        self.out2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        # Layer 1 with residual
        h1 = self.out1(self.L1(x) * self.R1(x))
        r1 = x + h1  # residual stream

        # Layer 2 with residual (output is final)
        h2 = self.out2(self.L2(r1) * self.R2(r1))
        return h2

    def weight_tensor_l1(self):
        return torch.einsum("hi,hj,kh->ijk", self.L1.weight, self.R1.weight, self.out1.weight)

    def weight_tensor_l2(self):
        return torch.einsum("hi,hj,kh->ijk", self.L2.weight, self.R2.weight, self.out2.weight)


def make_two_layer_and_dataset():
    """6 boolean inputs, 2 composed AND-gate outputs.
    L1: h0 = x0 & x1, h1 = x2 & x3, h2 = x4 & x5
    L2: z0 = h0 & h1, z1 = h1 & h2
    """
    inputs = torch.tensor(list(itertools.product([0, 1], repeat=6)), dtype=torch.float32)
    h0 = inputs[:, 0] * inputs[:, 1]
    h1 = inputs[:, 2] * inputs[:, 3]
    h2 = inputs[:, 4] * inputs[:, 5]
    targets = torch.stack([h0 * h1, h1 * h2], dim=1)
    return inputs, targets


def train_two_layer(d_hidden=16, n_epochs=10000, lr=1e-3):
    """Train two-layer bilinear on composed AND-gate task."""
    inputs, targets = make_two_layer_and_dataset()
    model = TwoLayerBilinearResidual(d_in=6, d_hidden=d_hidden, d_out=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        pred = model(inputs)
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2000 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    with torch.no_grad():
        pred = model(inputs)
        acc = ((pred > 0.5).float() == targets).float().mean().item()
        print(f"  Final accuracy: {acc:.4f}")

    return model


def run_stage4():
    """Stage 4: Two-layer stacked bilinear."""
    print("\n" + "=" * 60)
    print("STAGE 4: Two-Layer Stacked Bilinear with Residual")
    print("=" * 60)

    print("\nTraining two-layer model...")
    model = train_two_layer(d_hidden=16)

    # Tucker for Layer 1
    W1 = model.weight_tensor_l1().detach()
    print(f"\nL1 weight tensor shape: {W1.shape}, norm: {W1.norm():.4f}")

    print("\n--- L1 rank search ---")
    for r in range(1, 6):
        _, _, _, _, m = tucker_pipeline(W1, r)
        print(f"  rank={r}: recon={m['recon_error']:.6f}, odm={m['odm_after']:.4f}")

    G1_rot, A1_rot, B1_rot, C1_rot, m1 = tucker_pipeline(W1, 3)
    print(f"\nL1 at rank=3: recon={m1['recon_error']:.6f}, odm={m1['odm_after']:.4f}")

    feature_labels = ["x0", "x1", "x2", "x3", "x4", "x5"]
    print(f"  L1 factor directions:")
    for c in range(3):
        left = feature_labels[A1_rot[:, c].abs().argmax().item()]
        right = feature_labels[B1_rot[:, c].abs().argmax().item()]
        print(f"    Component {c}: {left} x {right}")

    # Tucker for Layer 2
    W2 = model.weight_tensor_l2().detach()
    print(f"\nL2 weight tensor shape: {W2.shape}, norm: {W2.norm():.4f}")

    print("\n--- L2 rank search ---")
    for r in range(1, 5):
        _, _, _, _, m = tucker_pipeline(W2, r)
        print(f"  rank={r}: recon={m['recon_error']:.6f}, odm={m['odm_after']:.4f}")

    G2_rot, A2_rot, B2_rot, C2_rot, m2 = tucker_pipeline(W2, 2)
    print(f"\nL2 at rank=2: recon={m2['recon_error']:.6f}, odm={m2['odm_after']:.4f}")

    # Inter-layer alignment
    print("\n--- Inter-layer alignment ---")
    overlap_left = C1_rot.T @ A2_rot   # (3, 2)
    overlap_right = C1_rot.T @ B2_rot  # (3, 2)
    print(f"  Overlap (C1^T @ A2):\n{overlap_left.detach().numpy()}")
    print(f"  Overlap (C1^T @ B2):\n{overlap_right.detach().numpy()}")

    # Ghost circuit check on dataset
    print("\n--- Ghost circuit check (L2) ---")
    inputs, _ = make_two_layer_and_dataset()
    with torch.no_grad():
        h1 = model.out1(model.L1(inputs) * model.R1(inputs))
        residual = inputs + h1  # what L2 sees

    threshold = 0.5
    for p in range(A2_rot.shape[1]):
        for q in range(B2_rot.shape[1]):
            left_act = (residual @ A2_rot[:, p]).abs()
            right_act = (residual @ B2_rot[:, q]).abs()
            co_active = ((left_act > threshold) & (right_act > threshold)).float().mean().item()
            g_val = G2_rot[p, q, :].abs().max().item() if p < G2_rot.shape[0] and q < G2_rot.shape[1] else 0
            ghost = "GHOST" if co_active < 0.01 and g_val > 0.1 else ""
            print(f"  G2[{p},{q},:] max={g_val:.3f}, co-occurrence={co_active:.3f} {ghost}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # L1 core
    for r_idx in range(min(3, G1_rot.shape[2])):
        if r_idx < len(axes) - 1:
            pass  # skip individual slices, show summary
    ax = axes[0]
    diag1 = [G1_rot[i, i, i].item() for i in range(3)]
    ax.bar(range(3), diag1)
    ax.set_title("L1 core diagonal")
    ax.set_xlabel("Component")

    # L2 core
    ax = axes[1]
    diag2 = [G2_rot[i, i, i].item() for i in range(2)]
    ax.bar(range(2), diag2)
    ax.set_title("L2 core diagonal")
    ax.set_xlabel("Component")

    # Overlap
    ax = axes[2]
    overlap_combined = torch.cat([overlap_left.abs(), overlap_right.abs()], dim=1)
    im = ax.imshow(overlap_combined.detach().numpy(), cmap="Blues", aspect="auto")
    ax.set_title("Inter-layer overlap |C1^T @ [A2, B2]|")
    ax.set_xlabel("L2 components (left | right)")
    ax.set_ylabel("L1 components")
    plt.colorbar(im, ax=ax)

    plt.suptitle("Stage 4: Two-layer Tucker decomposition")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage4_two_layer.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage4_two_layer.png'}")

    return model


# %% Main

def main():
    stage1_results = run_stage1()
    run_stage1_rank_search()
    model2, G2, A2, B2, C2, metrics2 = run_stage2()
    model3 = run_stage3()
    model4 = run_stage4()

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

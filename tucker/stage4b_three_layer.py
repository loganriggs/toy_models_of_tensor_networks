"""Stage 4b: Three-layer stacked bilinear with residual.

Compares rotation strategies for multi-layer Tucker decomposition:
1. Independent: rotate each layer independently (fastest, no cross-layer info)
2. Bottom-up: rotate L1 first, then constrain L2's input rotation to align
   with L1's output rotation, then L3 aligns with L2
3. Joint: optimize all rotations together to minimize total off-diagonal mass
   plus inter-layer overlap sparsity

Task: 3 layers of composed AND-gates over 8 boolean inputs.
  L1: h0 = x0 & x1, h1 = x2 & x3, h2 = x4 & x5, h3 = x6 & x7
  L2: g0 = h0 & h1, g1 = h2 & h3
  L3: z0 = g0 & g1

Run: python tucker/stage4b_three_layer.py
"""

# %% Imports

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tucker_pipeline import tucker_pipeline, hosvd, rotate_core, off_diagonal_mass, find_sparse_rotation

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)


# %% Three-layer model

class ThreeLayerBilinearResidual(nn.Module):
    """Three-layer bilinear with residual stream."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.L1 = nn.Linear(d_in, d_hidden, bias=False)
        self.R1 = nn.Linear(d_in, d_hidden, bias=False)
        self.out1 = nn.Linear(d_hidden, d_in, bias=False)

        self.L2 = nn.Linear(d_in, d_hidden, bias=False)
        self.R2 = nn.Linear(d_in, d_hidden, bias=False)
        self.out2 = nn.Linear(d_hidden, d_in, bias=False)

        self.L3 = nn.Linear(d_in, d_hidden, bias=False)
        self.R3 = nn.Linear(d_in, d_hidden, bias=False)
        self.out3 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        h1 = self.out1(self.L1(x) * self.R1(x))
        r1 = x + h1

        h2 = self.out2(self.L2(r1) * self.R2(r1))
        r2 = r1 + h2

        h3 = self.out3(self.L3(r2) * self.R3(r2))
        return h3

    def weight_tensor(self, layer):
        """Extract W[i,j,k] = sum_h L[h,i]*R[h,j]*out[k,h] for given layer."""
        L = getattr(self, f"L{layer}").weight
        R = getattr(self, f"R{layer}").weight
        O = getattr(self, f"out{layer}").weight
        return torch.einsum("hi,hj,kh->ijk", L, R, O)

    def residual_at(self, x, after_layer):
        """Compute residual stream after given layer."""
        r = x
        for l in range(1, after_layer + 1):
            L = getattr(self, f"L{l}")
            R = getattr(self, f"R{l}")
            O = getattr(self, f"out{l}")
            h = O(L(r) * R(r))
            r = r + h
        return r


# %% Task

def make_three_layer_dataset():
    """8 boolean inputs, 1 composed AND-gate output.
    L1: h0 = x0 & x1, h1 = x2 & x3, h2 = x4 & x5, h3 = x6 & x7
    L2: g0 = h0 & h1, g1 = h2 & h3
    L3: z0 = g0 & g1
    """
    inputs = torch.tensor(list(itertools.product([0, 1], repeat=8)), dtype=torch.float32)
    h0 = inputs[:, 0] * inputs[:, 1]
    h1 = inputs[:, 2] * inputs[:, 3]
    h2 = inputs[:, 4] * inputs[:, 5]
    h3 = inputs[:, 6] * inputs[:, 7]
    g0 = h0 * h1
    g1 = h2 * h3
    z0 = g0 * g1
    targets = z0.unsqueeze(1)
    return inputs, targets


def train_three_layer(d_hidden=32, n_epochs=15000, lr=1e-3):
    inputs, targets = make_three_layer_dataset()
    model = ThreeLayerBilinearResidual(d_in=8, d_hidden=d_hidden, d_out=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        pred = model(inputs)
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 3000 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    with torch.no_grad():
        pred = model(inputs)
        acc = ((pred > 0.5).float() == targets).float().mean().item()
        print(f"  Final accuracy: {acc:.4f}")
    return model


# %% Strategy 1: Independent layer-wise Tucker

def independent_tucker(model, ranks):
    """Decompose each layer independently. No cross-layer alignment."""
    results = {}
    for layer in range(1, 4):
        W = model.weight_tensor(layer).detach()
        r = ranks[layer - 1]
        G_rot, A_rot, B_rot, C_rot, metrics = tucker_pipeline(W, r)
        results[layer] = {
            "G": G_rot, "A": A_rot, "B": B_rot, "C": C_rot,
            "metrics": metrics, "rank": r,
        }
    return results


# %% Strategy 2: Bottom-up aligned Tucker

def bottom_up_tucker(model, ranks):
    """Decompose bottom-up: L1 freely, then L2 input aligned to L1 output, etc.

    After Tucker on L1, the output factor C1_rot defines a basis for L1's output space.
    For L2, we constrain the input factors (A2, B2) to partially align with C1_rot
    by doing a joint rotation of C1 and A2/B2 after independent Tucker.
    """
    results = {}

    # Layer 1: free Tucker
    W1 = model.weight_tensor(1).detach()
    G1, A1, B1, C1, m1 = tucker_pipeline(W1, ranks[0])
    results[1] = {"G": G1, "A": A1, "B": B1, "C": C1, "metrics": m1}

    # Layer 2: Tucker, then align input factors with L1 output
    W2 = model.weight_tensor(2).detach()
    G2_raw, (A2_h, B2_h, C2_h) = hosvd(W2, (ranks[1], ranks[1], ranks[1]))

    # Find rotation for L2 that also minimizes overlap with C1
    # Step 1: rotate L2 core for superdiagonality
    R2_A, R2_B, R2_C = find_sparse_rotation(G2_raw, n_restarts=10, n_sweeps=50)
    G2 = rotate_core(G2_raw, R2_A, R2_B, R2_C)
    A2 = A2_h @ R2_A.T
    B2 = B2_h @ R2_B.T
    C2 = C2_h @ R2_C.T

    # Step 2: joint align C1 output with A2, B2 input
    # Find rotation Q that minimizes ||C1 @ Q - A2||_F + ||C1 @ Q - B2||_F
    # This is a Procrustes problem: min ||C1 Q - target||
    # We use the average of A2 and B2 as target (they both should align with C1's output)
    # Only align the first min(r_C1, r_A2) columns
    r_align = min(C1.shape[1], A2.shape[1])
    target = (A2[:, :r_align] + B2[:, :r_align]) / 2
    # Procrustes: Q = U @ Vh from SVD(C1^T @ target)
    M = C1[:, :r_align].T @ target
    U, _, Vh = torch.linalg.svd(M)
    Q = U @ Vh
    # Apply to C1 only (don't re-rotate L2, just record the alignment quality)
    C1_aligned = C1[:, :r_align] @ Q

    overlap_before_left = C1[:, :r_align].T @ A2[:, :r_align]
    overlap_after_left = C1_aligned.T @ A2[:, :r_align]

    m2 = {
        "recon_error": 0.0,  # not recomputed
        "odm_after": off_diagonal_mass(G2),
        "odm_before": off_diagonal_mass(G2_raw),
        "ranks": (ranks[1], ranks[1], ranks[1]),
    }
    results[2] = {
        "G": G2, "A": A2, "B": B2, "C": C2, "metrics": m2,
        "overlap_before": overlap_before_left,
        "overlap_after": overlap_after_left,
        "C1_aligned": C1_aligned,
    }

    # Layer 3: same approach, align with L2 output
    W3 = model.weight_tensor(3).detach()
    G3_raw, (A3_h, B3_h, C3_h) = hosvd(W3, (ranks[2], ranks[2], ranks[2]))
    R3_A, R3_B, R3_C = find_sparse_rotation(G3_raw, n_restarts=10, n_sweeps=50)
    G3 = rotate_core(G3_raw, R3_A, R3_B, R3_C)
    A3 = A3_h @ R3_A.T
    B3 = B3_h @ R3_B.T
    C3 = C3_h @ R3_C.T

    r_align2 = min(C2.shape[1], A3.shape[1])
    target2 = (A3[:, :r_align2] + B3[:, :r_align2]) / 2
    M2 = C2[:, :r_align2].T @ target2
    U2, _, Vh2 = torch.linalg.svd(M2)
    Q2 = U2 @ Vh2
    C2_aligned = C2[:, :r_align2] @ Q2

    overlap_L2_L3_before = C2[:, :r_align2].T @ A3[:, :r_align2]
    overlap_L2_L3_after = C2_aligned.T @ A3[:, :r_align2]

    m3 = {
        "recon_error": 0.0,
        "odm_after": off_diagonal_mass(G3),
        "odm_before": off_diagonal_mass(G3_raw),
        "ranks": (ranks[2], ranks[2], ranks[2]),
    }
    results[3] = {
        "G": G3, "A": A3, "B": B3, "C": C3, "metrics": m3,
        "overlap_before": overlap_L2_L3_before,
        "overlap_after": overlap_L2_L3_after,
        "C_prev_aligned": C2_aligned,
    }

    return results


# %% Strategy 3: Joint optimization

def joint_tucker(model, ranks, lambda_overlap=0.5, n_iters=200):
    """Jointly optimize all layer rotations to minimize:
    L = sum_l ||G_l_rot - diag||^2 + lambda * sum_l ||C_l^T @ A_{l+1}||_1

    We first do independent HOSVD, then jointly optimize the rotations
    using alternating Givens sweeps with the combined objective.
    """
    # Step 1: Independent HOSVD for all layers
    layers_data = {}
    for layer in range(1, 4):
        W = model.weight_tensor(layer).detach()
        r = ranks[layer - 1]
        G, (A, B, C) = hosvd(W, (r, r, r))
        layers_data[layer] = {"G": G, "A": A, "B": B, "C": C}

    # Step 2: Initialize rotations as identity
    rotations = {}
    for layer in range(1, 4):
        r = ranks[layer - 1]
        rotations[layer] = {
            "R_A": torch.eye(r),
            "R_B": torch.eye(r),
            "R_C": torch.eye(r),
        }

    def compute_objective():
        """Compute combined objective: core diagonality + inter-layer sparsity."""
        total = 0.0
        for layer in range(1, 4):
            G = layers_data[layer]["G"]
            R_A = rotations[layer]["R_A"]
            R_B = rotations[layer]["R_B"]
            R_C = rotations[layer]["R_C"]
            G_rot = rotate_core(G, R_A, R_B, R_C)
            # Core diagonality: maximize sum_i G_rot[i,i,i]^2
            r = min(G_rot.shape)
            diag_energy = sum(G_rot[i, i, i] ** 2 for i in range(r))
            total_energy = (G_rot ** 2).sum()
            total += (total_energy - diag_energy)  # off-diagonal energy

        # Inter-layer overlap penalty
        for layer in range(1, 3):
            A_h = layers_data[layer]["A"]
            C_h = layers_data[layer]["C"]
            A_next_h = layers_data[layer + 1]["A"]
            B_next_h = layers_data[layer + 1]["B"]

            R_C = rotations[layer]["R_C"]
            R_A_next = rotations[layer + 1]["R_A"]
            R_B_next = rotations[layer + 1]["R_B"]

            C_rot = C_h @ R_C.T
            A_next_rot = A_next_h @ R_A_next.T
            B_next_rot = B_next_h @ R_B_next.T

            r_align = min(C_rot.shape[1], A_next_rot.shape[1])
            overlap_left = C_rot[:, :r_align].T @ A_next_rot[:, :r_align]
            overlap_right = C_rot[:, :r_align].T @ B_next_rot[:, :r_align]

            # L1 penalty on off-diagonal elements of overlap
            for i in range(r_align):
                for j in range(r_align):
                    if i != j:
                        total += lambda_overlap * (overlap_left[i, j].abs() + overlap_right[i, j].abs())

        return total

    def _givens(n, i, j, theta):
        R = torch.eye(n)
        c, s = torch.cos(theta), torch.sin(theta)
        R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
        return R

    # Step 3: Alternating Givens sweeps over all rotations
    best_obj = float("inf")
    best_rotations = None

    for restart in range(5):
        if restart > 0:
            for layer in range(1, 4):
                r = ranks[layer - 1]
                rotations[layer] = {
                    "R_A": torch.linalg.qr(torch.randn(r, r))[0],
                    "R_B": torch.linalg.qr(torch.randn(r, r))[0],
                    "R_C": torch.linalg.qr(torch.randn(r, r))[0],
                }

        for sweep in range(n_iters):
            for layer in range(1, 4):
                r = ranks[layer - 1]
                for mode in ["R_A", "R_B", "R_C"]:
                    for i in range(r):
                        for j in range(i + 1, r):
                            best_theta = 0.0
                            best_val = compute_objective()
                            for theta in torch.linspace(-torch.pi, torch.pi, 24):
                                Rij = _givens(r, i, j, theta)
                                old_R = rotations[layer][mode].clone()
                                rotations[layer][mode] = old_R @ Rij
                                val = compute_objective()
                                if val < best_val:
                                    best_val = val
                                    best_theta = theta
                                rotations[layer][mode] = old_R
                            if best_theta != 0.0:
                                Rij = _givens(r, i, j, best_theta)
                                rotations[layer][mode] = rotations[layer][mode] @ Rij

        obj = compute_objective()
        if obj < best_obj:
            best_obj = obj
            best_rotations = {
                l: {k: v.clone() for k, v in rots.items()}
                for l, rots in rotations.items()
            }

    # Apply best rotations
    results = {}
    for layer in range(1, 4):
        G = layers_data[layer]["G"]
        A = layers_data[layer]["A"]
        B = layers_data[layer]["B"]
        C = layers_data[layer]["C"]
        R_A = best_rotations[layer]["R_A"]
        R_B = best_rotations[layer]["R_B"]
        R_C = best_rotations[layer]["R_C"]
        G_rot = rotate_core(G, R_A, R_B, R_C)
        A_rot = A @ R_A.T
        B_rot = B @ R_B.T
        C_rot = C @ R_C.T
        m = {
            "odm_after": off_diagonal_mass(G_rot),
            "odm_before": off_diagonal_mass(G),
            "recon_error": 0.0,
            "ranks": (ranks[layer - 1],) * 3,
        }
        results[layer] = {"G": G_rot, "A": A_rot, "B": B_rot, "C": C_rot, "metrics": m}

    return results


# %% Analysis and comparison

def analyze_results(results, model, label, feature_labels):
    """Print analysis for a rotation strategy."""
    print(f"\n{'='*60}")
    print(f"Strategy: {label}")
    print(f"{'='*60}")

    for layer in range(1, 4):
        res = results[layer]
        G = res["G"]
        A = res["A"]
        B = res["B"]
        r = min(G.shape)

        odm = off_diagonal_mass(G)
        diag = [G[i, i, i].item() for i in range(r)]

        print(f"\n  Layer {layer}: rank={r}, ODM={odm:.4f}")
        print(f"    Core diagonal: {[f'{d:.3f}' for d in diag]}")

        print(f"    Factor directions:")
        for c in range(r):
            left = feature_labels[A[:, c].abs().argmax().item()] if A.shape[0] <= len(feature_labels) else f"dim{A[:, c].abs().argmax().item()}"
            right = feature_labels[B[:, c].abs().argmax().item()] if B.shape[0] <= len(feature_labels) else f"dim{B[:, c].abs().argmax().item()}"
            print(f"      Component {c}: {left} x {right}")

    # Inter-layer overlap
    for layer in range(1, 3):
        C_curr = results[layer]["C"]
        A_next = results[layer + 1]["A"]
        B_next = results[layer + 1]["B"]
        r_align = min(C_curr.shape[1], A_next.shape[1])
        overlap_L = C_curr[:, :r_align].T @ A_next[:, :r_align]
        overlap_R = C_curr[:, :r_align].T @ B_next[:, :r_align]

        # Sparsity = fraction of entries < 0.1
        all_entries = torch.cat([overlap_L.abs().flatten(), overlap_R.abs().flatten()])
        sparsity = (all_entries < 0.1).float().mean().item()
        off_diag = 0.0
        for i in range(r_align):
            for j in range(r_align):
                if i != j:
                    off_diag += overlap_L[i, j].abs().item() + overlap_R[i, j].abs().item()
        n_off = 2 * r_align * (r_align - 1)
        mean_off = off_diag / n_off if n_off > 0 else 0

        print(f"\n  L{layer}->L{layer+1} overlap: sparsity={sparsity:.2f}, mean_off_diag={mean_off:.4f}")
        print(f"    C{layer}^T @ A{layer+1}:\n      {overlap_L.detach().numpy()}")
        print(f"    C{layer}^T @ B{layer+1}:\n      {overlap_R.detach().numpy()}")

    # Total off-diagonal mass across all layers
    total_odm = sum(off_diagonal_mass(results[l]["G"]) for l in range(1, 4)) / 3
    print(f"\n  Average ODM across layers: {total_odm:.4f}")

    return total_odm


def plot_comparison(all_results, feature_labels):
    """Visualize comparison of the three strategies."""
    strategies = list(all_results.keys())
    n_strat = len(strategies)

    # Task labels per layer
    layer_tasks = {
        1: ["x0&x1", "x2&x3", "x4&x5", "x6&x7"],
        2: ["h0&h1", "h2&h3"],
        3: ["g0&g1"],
    }

    fig, axes = plt.subplots(3, n_strat, figsize=(5 * n_strat, 12))

    for col, strat_name in enumerate(strategies):
        results = all_results[strat_name]

        for row, layer in enumerate(range(1, 4)):
            ax = axes[row, col]
            G = results[layer]["G"]
            A = results[layer]["A"]
            r = min(G.shape)

            # Auto-assign components to tasks via output factor
            C = results[layer]["C"]
            comp_labels = []
            tasks = layer_tasks[layer]
            for c in range(r):
                top_out = C[:, c].abs().argmax().item()
                if top_out < len(tasks):
                    comp_labels.append(tasks[top_out])
                else:
                    comp_labels.append(f"C{c}")

            diag_vals = [G[i, i, i].item() for i in range(r)]
            off_diag_total = (G ** 2).sum().item() - sum(d ** 2 for d in diag_vals)

            bars = ax.bar(range(r), [abs(d) for d in diag_vals], color="steelblue", label="diagonal")
            avg_off = (abs(off_diag_total) / max(r * (r - 1), 1)) ** 0.5
            ax.axhline(y=avg_off, color="red",
                       linestyle="--", label="avg off-diag", alpha=0.7)

            odm = off_diagonal_mass(G)
            ax.set_title(f"{strat_name} L{layer} (ODM={odm:.3f})", fontsize=10)
            ax.set_xticks(range(r))
            ax.set_xticklabels(comp_labels, fontsize=8, rotation=30)
            ax.set_ylabel("|G[i,i,i]|")
            if row == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage4b_strategy_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'stage4b_strategy_comparison.png'}")

    # Overlap matrices comparison
    fig, axes = plt.subplots(2, n_strat, figsize=(5 * n_strat, 8))
    for col, strat_name in enumerate(strategies):
        results = all_results[strat_name]
        for row, (l1, l2) in enumerate([(1, 2), (2, 3)]):
            ax = axes[row, col]
            C = results[l1]["C"]
            A_next = results[l2]["A"]
            B_next = results[l2]["B"]
            r = min(C.shape[1], A_next.shape[1])
            overlap = torch.cat([
                (C[:, :r].T @ A_next[:, :r]).abs(),
                (C[:, :r].T @ B_next[:, :r]).abs(),
            ], dim=1)
            im = ax.imshow(overlap.detach().numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=1)
            ax.set_title(f"{strat_name} L{l1}->L{l2}", fontsize=10)

            # Label axes with component indices
            tasks_curr = layer_tasks[l1]
            tasks_next = layer_tasks[l2]
            ax.set_yticks(range(r))
            ax.set_yticklabels([f"C{i}" for i in range(r)], fontsize=9)
            ax.set_ylabel(f"L{l1} output comp")

            # X-axis: [A_0..A_r | B_0..B_r]
            x_labels = [f"A{i}" for i in range(r)] + [f"B{i}" for i in range(r)]
            ax.set_xticks(range(2 * r))
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_xlabel(f"L{l2} input comps [left|right]")

            # Add vertical separator between A and B
            ax.axvline(x=r - 0.5, color="white", linewidth=2)

            # Annotate cells
            data = overlap.detach().numpy()
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    if val > 0.3:
                        color = "white" if val > 0.6 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                               color=color, fontsize=8)
            plt.colorbar(im, ax=ax)

    plt.suptitle("Cross-layer overlap: |C_l^T @ A_{l+1}| and |C_l^T @ B_{l+1}|\n"
                 "Ideal: diagonal (each output comp feeds one input comp)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage4b_overlap_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'stage4b_overlap_comparison.png'}")


# %% Main

def main():
    print("=" * 60)
    print("STAGE 4b: Three-Layer Stacked Bilinear")
    print("=" * 60)

    print("\nTraining three-layer model...")
    model = train_three_layer(d_hidden=32)

    # Print weight tensor info
    for layer in range(1, 4):
        W = model.weight_tensor(layer).detach()
        print(f"  L{layer} weight tensor: shape={W.shape}, norm={W.norm():.4f}")

    feature_labels = [f"x{i}" for i in range(8)]
    ranks = [4, 2, 1]  # true ranks: L1=4 AND-gates, L2=2, L3=1

    # Strategy 1: Independent
    print("\n--- Running independent Tucker ---")
    results_indep = independent_tucker(model, ranks)

    # Strategy 2: Bottom-up
    print("\n--- Running bottom-up Tucker ---")
    results_bottomup = bottom_up_tucker(model, ranks)

    # Strategy 3: Joint (expensive — reduce iterations for feasibility)
    print("\n--- Running joint Tucker (this takes a while) ---")
    results_joint = joint_tucker(model, ranks, lambda_overlap=0.3, n_iters=30)

    # Analyze all
    odm_indep = analyze_results(results_indep, model, "Independent", feature_labels)
    odm_bottomup = analyze_results(results_bottomup, model, "Bottom-up", feature_labels)
    odm_joint = analyze_results(results_joint, model, "Joint", feature_labels)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Strategy':<15} {'Avg ODM':>10}")
    print(f"  {'Independent':<15} {odm_indep:>10.4f}")
    print(f"  {'Bottom-up':<15} {odm_bottomup:>10.4f}")
    print(f"  {'Joint':<15} {odm_joint:>10.4f}")

    all_results = {
        "Independent": results_indep,
        "Bottom-up": results_bottomup,
        "Joint": results_joint,
    }
    plot_comparison(all_results, feature_labels)

    # Ghost circuit check on L3
    print("\n--- Ghost circuit check (L3) ---")
    inputs, _ = make_three_layer_dataset()
    with torch.no_grad():
        r2 = model.residual_at(inputs, 2)
    A3 = results_indep[3]["A"]
    B3 = results_indep[3]["B"]
    G3 = results_indep[3]["G"]
    threshold = 0.3
    for p in range(A3.shape[1]):
        for q in range(B3.shape[1]):
            left_act = (r2 @ A3[:, p]).abs()
            right_act = (r2 @ B3[:, q]).abs()
            co_active = ((left_act > threshold) & (right_act > threshold)).float().mean().item()
            g_val = G3[p, q, :].abs().max().item() if p < G3.shape[0] and q < G3.shape[1] else 0
            ghost = "GHOST" if co_active < 0.01 and g_val > 0.05 else ""
            print(f"  G3[{p},{q},:] max={g_val:.3f}, co-occurrence={co_active:.3f} {ghost}")


if __name__ == "__main__":
    main()

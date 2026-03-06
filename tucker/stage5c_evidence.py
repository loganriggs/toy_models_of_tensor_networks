"""Stage 5c: Comprehensive evidence for per-head circuit claims.

Generates labeled figures showing:
1. Per-head QK and OV matrices with token labels at key distances
2. Tucker factor interpretations with labeled bar charts
3. L0 -> L1 composition scores (V-composition and K-composition)
4. Logit attribution: how much each component contributes to each rule
5. Coverage analysis: what % of model performance is explained

Run: python tucker/stage5c_evidence.py
"""

# %% Imports

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "attn_circuits"))

from model import BilinearGPT, Config
from generator_v3 import LanguageV3, make_batch, per_rule_loss, true_entropies
from stage5b_position_conditioned import (
    load_model, get_token_labels, extract_head_weights,
    build_circuit_tensor_at_distance, rope_rotation_matrix,
)

FIGURES_DIR = Path(__file__).parent / "figures" / "stage5c"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# %% Helpers

def get_category_order():
    """Return category order and sorted token indices."""
    labels, categories = get_token_labels()
    cat_order = ["NOUN", "PLACE", "VERB_T", "VERB_I", "ADJ", "FUNC",
                 "PUNCT", "OPEN", "CLOSE", "STRUCT"]
    sorted_indices = sorted(
        range(len(labels)),
        key=lambda i: (cat_order.index(categories.get(i, "STRUCT")), labels[i])
    )
    sorted_labels = [labels[i] for i in sorted_indices]
    # Category boundaries for drawing lines on heatmaps
    boundaries = []
    prev_cat = None
    for pos, idx in enumerate(sorted_indices):
        cat = categories.get(idx, "STRUCT")
        if cat != prev_cat:
            boundaries.append((pos, cat))
            prev_cat = cat
    return labels, categories, sorted_indices, sorted_labels, boundaries


def labeled_heatmap(ax, mat, row_labels, col_labels, boundaries, title,
                    cmap="RdBu_r", symmetric=True, fontsize=5):
    """Plot a heatmap with token labels and category boundary lines."""
    if symmetric:
        vmax = mat.abs().max().item()
        if vmax < 1e-8:
            vmax = 1.0
        im = ax.imshow(mat.numpy(), cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(mat.numpy(), cmap=cmap, aspect="auto")
    ax.set_title(title, fontsize=9)

    # Category boundary lines
    for pos, cat in boundaries:
        ax.axhline(pos - 0.5, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(pos - 0.5, color="gray", linewidth=0.5, alpha=0.5)

    # Tick labels (show every token)
    n = len(row_labels)
    if n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(col_labels, fontsize=fontsize, rotation=90)
        ax.set_yticks(range(n))
        ax.set_yticklabels(row_labels, fontsize=fontsize)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# %% 1. Per-head labeled QK and OV matrices

def filter_matrix(mat, row_labels, col_labels, threshold_frac=0.05):
    """Remove rows/cols where max |value| < threshold_frac * global max.

    Returns filtered matrix, labels, and original indices.
    """
    vmax = mat.abs().max().item()
    if vmax < 1e-8:
        return mat, row_labels, col_labels, list(range(len(row_labels))), list(range(len(col_labels)))

    thresh = threshold_frac * vmax

    # Keep rows where at least one entry exceeds threshold
    row_mask = mat.abs().max(dim=1).values > thresh
    col_mask = mat.abs().max(dim=0).values > thresh

    row_indices = [i for i in range(len(row_labels)) if row_mask[i]]
    col_indices = [i for i in range(len(col_labels)) if col_mask[i]]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return mat, row_labels, col_labels, list(range(len(row_labels))), list(range(len(col_labels)))

    filtered = mat[row_mask][:, col_mask]
    filt_rows = [row_labels[i] for i in row_indices]
    filt_cols = [col_labels[i] for i in col_indices]
    return filtered, filt_rows, filt_cols, row_indices, col_indices


def filtered_heatmap(ax, mat, row_labels, col_labels, title,
                     cmap="RdBu_r", symmetric=True, fontsize=6,
                     threshold_frac=0.05):
    """Plot a heatmap with near-zero rows/cols removed."""
    filt, fr, fc, _, _ = filter_matrix(mat, row_labels, col_labels, threshold_frac)

    if symmetric:
        vmax = filt.abs().max().item()
        if vmax < 1e-8:
            vmax = 1.0
        im = ax.imshow(filt.numpy(), cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(filt.numpy(), cmap=cmap, aspect="auto")

    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(len(fc)))
    ax.set_xticklabels(fc, fontsize=fontsize, rotation=90)
    ax.set_yticks(range(len(fr)))
    ax.set_yticklabels(fr, fontsize=fontsize)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_head_evidence(model, layer_idx, head_idx, labels, categories,
                       sorted_indices, sorted_labels, boundaries,
                       distances=[1, 3, 8]):
    """Plot filtered QK_d patterns + single OV for one head.

    QK(d) tells you: "when the query is token q and the key is token k at
    distance d positions back, how strongly does this head attend?"
    Higher value = more attention weight on that (query, key) pair.

    OV tells you: "when this head attends to key token k, what output logit
    does it produce?" It maps key tokens to output token predictions.
    OV is position-independent (same for all d).

    Reading the full circuit: T[q,k,o] = QK(d)[q,k] * OV[k,o]
    "When q attends to k (at distance d), produce output o."
    """
    head_label = f"L{layer_idx}H{head_idx}"
    n_dist = len(distances)

    # Layout: top row = QK at each distance, bottom row = single OV (wider)
    fig = plt.figure(figsize=(6 * n_dist, 14))
    gs = fig.add_gridspec(2, n_dist, height_ratios=[1, 1], hspace=0.35)

    with torch.no_grad():
        # Get OV once (same for all distances)
        _, _, _, _, OV = build_circuit_tensor_at_distance(
            model, layer_idx, head_idx, 1
        )
        OV_sorted = OV[sorted_indices][:, sorted_indices]

        for col, d in enumerate(distances):
            _, _, _, QK_d, _ = build_circuit_tensor_at_distance(
                model, layer_idx, head_idx, d
            )
            QK_sorted = QK_d[sorted_indices][:, sorted_indices]

            # QK pattern (filtered)
            ax = fig.add_subplot(gs[0, col])
            filtered_heatmap(ax, QK_sorted, sorted_labels, sorted_labels,
                             f"QK (d={d}): which query attends to which key?")
            ax.set_ylabel("query token (current position)")
            ax.set_xlabel("key token (d positions back)")

    # OV circuit (single, spanning full width)
    ax_ov = fig.add_subplot(gs[1, :])
    filtered_heatmap(ax_ov, OV_sorted, sorted_labels, sorted_labels,
                     "OV: when attending to key token k, what output does this head produce?")
    ax_ov.set_ylabel("key (attended) token")
    ax_ov.set_xlabel("output token logit")

    fig.suptitle(f"{head_label}", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(FIGURES_DIR / f"evidence_{head_label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: evidence_{head_label}.png")


# %% 2. Tucker factor bar charts

def plot_tucker_factors(model, layer_idx, head_idx, d, labels, categories):
    """Show Tucker factor columns as labeled bar charts for one head at distance d."""
    from tucker_pipeline import tucker_pipeline
    head_label = f"L{layer_idx}H{head_idx}"

    with torch.no_grad():
        T_d, _, _, _, _ = build_circuit_tensor_at_distance(
            model, layer_idx, head_idx, d
        )

    # Auto rank
    rank = 4
    for r in range(1, 5):
        try:
            _, _, _, _, m = tucker_pipeline(T_d, r, fast=True)
            if m["recon_error"] < 0.15:
                rank = r
                break
        except Exception:
            continue

    G_rot, A_rot, B_rot, C_rot, metrics = tucker_pipeline(T_d, rank, fast=True)
    r = min(rank, *G_rot.shape)

    fig, axes = plt.subplots(r, 3, figsize=(18, 3.5 * r))
    if r == 1:
        axes = axes[None, :]

    for c in range(r):
        g_val = G_rot[c, c, c].item()

        for mode_idx, (factor, factor_name) in enumerate(
            [(A_rot, "Query"), (B_rot, "Key"), (C_rot, "Output")]
        ):
            ax = axes[c, mode_idx]
            col = factor[:, c]
            colors = []
            cat_colors = {
                "NOUN": "#1f77b4", "PLACE": "#ff7f0e", "VERB_T": "#2ca02c",
                "VERB_I": "#d62728", "ADJ": "#9467bd", "FUNC": "#8c564b",
                "PUNCT": "#7f7f7f", "OPEN": "#bcbd22", "CLOSE": "#17becf",
                "STRUCT": "#aaaaaa",
            }
            for i in range(len(labels)):
                cat = categories.get(i, "STRUCT")
                colors.append(cat_colors.get(cat, "#aaaaaa"))

            ax.bar(range(len(labels)), col.numpy(), color=colors, width=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=5, rotation=90)
            ax.set_title(f"Component {c} {factor_name} (G[{c},{c},{c}]={g_val:.1f})",
                         fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5)

    plt.suptitle(f"{head_label} d={d}: Tucker factors (rank={rank}, "
                 f"recon={metrics['recon_error']:.3f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"factors_{head_label}_d{d}.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Saved: factors_{head_label}_d{d}.png")


# %% 3. L0 -> L1 composition

def compute_composition_scores(model):
    """Compute V-composition and K-composition between L0 and L1 heads.

    V-composition: How much does L1 read from L0's output?
      score = ||W_OV_L1_h2 @ W_OV_L0_h1||_F / (||W_OV_L1_h2||_F * ||W_OV_L0_h1||_F)

    K-composition: How much does L1's key computation use L0's output?
      score = ||W_K_L1_h2 @ W_O_L0_h1||_F / (||W_K_L1_h2||_F * ||W_O_L0_h1||_F)

    Q-composition: Similar for queries.
    """
    n_heads = model.config.n_head

    # Extract W_OV = V^T @ O^T for each head (maps residual -> residual via attention)
    def get_OV(layer, head):
        _, _, _, _, V, O = extract_head_weights(model, layer, head)
        return O @ V  # (n_embd, n_embd): residual -> residual

    def get_QK(layer, head, pair=1):
        Q1, K1, Q2, K2, _, _ = extract_head_weights(model, layer, head)
        if pair == 1:
            return K1, Q1  # K maps residual->head, Q maps residual->head
        else:
            return K2, Q2

    # V-composition: OV_L1 @ OV_L0
    v_comp = torch.zeros(n_heads, n_heads)
    # K-composition: K_L1 @ O_L0 (how L1's keys attend to L0's outputs)
    k_comp = torch.zeros(n_heads, n_heads)
    # Q-composition: Q_L1 @ O_L0 (how L1's queries use L0's outputs)
    q_comp = torch.zeros(n_heads, n_heads)

    with torch.no_grad():
        for h0 in range(n_heads):
            OV_L0 = get_OV(0, h0)  # (n_embd, n_embd)
            _, _, _, _, _, O_L0 = extract_head_weights(model, 0, h0)

            for h1 in range(n_heads):
                OV_L1 = get_OV(1, h1)
                K1_L1, Q1_L1 = get_QK(1, h1, pair=1)
                K2_L1, Q2_L1 = get_QK(1, h1, pair=2)

                # V-composition: ||OV_L1 @ OV_L0||_F normalized
                composed = OV_L1 @ OV_L0
                v_comp[h0, h1] = composed.norm() / (OV_L1.norm() * OV_L0.norm() + 1e-8)

                # K-composition: ||K_L1 @ O_L0||_F normalized (using K1)
                k_composed = K1_L1 @ O_L0
                k_comp[h0, h1] = k_composed.norm() / (K1_L1.norm() * O_L0.norm() + 1e-8)

                # Q-composition
                q_composed = Q1_L1 @ O_L0
                q_comp[h0, h1] = q_composed.norm() / (Q1_L1.norm() * O_L0.norm() + 1e-8)

    return v_comp, k_comp, q_comp


def plot_composition(model):
    """Plot V/K/Q composition matrices."""
    v_comp, k_comp, q_comp = compute_composition_scores(model)
    n_heads = model.config.n_head

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    head_labels = [f"L0H{h}" for h in range(n_heads)]
    head_labels_l1 = [f"L1H{h}" for h in range(n_heads)]

    for ax, mat, title in [
        (axes[0], v_comp, "V-composition\n(L1 reads L0 output)"),
        (axes[1], k_comp, "K-composition\n(L1 keys attend L0 output)"),
        (axes[2], q_comp, "Q-composition\n(L1 queries use L0 output)"),
    ]:
        im = ax.imshow(mat.numpy(), cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels(head_labels_l1, fontsize=9)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels(head_labels, fontsize=9)
        ax.set_xlabel("Layer 1 head")
        ax.set_ylabel("Layer 0 head")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)

        # Annotate with values
        for i in range(n_heads):
            for j in range(n_heads):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if mat[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "composition_L0_L1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: composition_L0_L1.png")

    return v_comp, k_comp, q_comp


# %% 4. Logit attribution per rule

def forward_with_cache(model, x):
    """Forward pass returning per-head attention outputs and the direct path.

    Since the model has no norms and no MLP, the decomposition is exact:
      logits = lm_head(embed + sum_h0 attn_L0_h(embed) + sum_h1 attn_L1_h(embed + sum_h0))

    Returns:
      - direct: embed contribution to logits  (B, T, V)
      - head_contribs: dict of (layer, head) -> contribution to logits (B, T, V)
      - full_logits: the actual model logits (B, T, V)
    """
    E = model.embed(x)  # (B, T, n_embd)
    B, T, C = E.shape
    n_heads = model.config.n_head

    # Layer 0: input is just embeddings
    block0 = model.blocks[0]
    attn0 = block0.attn
    D = attn0.head_dim

    # Full L0 attention computation with per-head outputs
    q1_0 = attn0.q1(E).view(B, T, n_heads, D)
    k1_0 = attn0.k1(E).view(B, T, n_heads, D)
    q2_0 = attn0.q2(E).view(B, T, n_heads, D)
    k2_0 = attn0.k2(E).view(B, T, n_heads, D)
    v_0 = attn0.v(E).view(B, T, n_heads, D)

    cos, sin = attn0.rotary(q1_0)
    from model import apply_rotary_emb
    q1_0 = apply_rotary_emb(q1_0, cos, sin)
    k1_0 = apply_rotary_emb(k1_0, cos, sin)
    q2_0 = apply_rotary_emb(q2_0, cos, sin)
    k2_0 = apply_rotary_emb(k2_0, cos, sin)

    scores1_0 = torch.einsum("bshd,bkhd->bhsk", q1_0, k1_0) / D
    scores2_0 = torch.einsum("bshd,bkhd->bhsk", q2_0, k2_0) / D
    pattern_0 = scores1_0 * scores2_0
    causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    pattern_0 = pattern_0.masked_fill(~causal, 0.0)

    # Per-head outputs for L0
    head_outputs_L0 = {}
    attn_full_0 = torch.zeros(B, T, C, device=x.device)
    for h in range(n_heads):
        y_h = torch.einsum("bsk,bkd->bsd", pattern_0[:, h], v_0[:, :, h])  # (B, T, D)
        out_h = y_h @ attn0.out.weight[:, h * D:(h + 1) * D].T  # (B, T, C)
        head_outputs_L0[h] = out_h
        attn_full_0 = attn_full_0 + out_h

    r0 = E + attn_full_0  # residual after L0

    # Layer 1: input is r0
    block1 = model.blocks[1]
    attn1 = block1.attn

    q1_1 = attn1.q1(r0).view(B, T, n_heads, D)
    k1_1 = attn1.k1(r0).view(B, T, n_heads, D)
    q2_1 = attn1.q2(r0).view(B, T, n_heads, D)
    k2_1 = attn1.k2(r0).view(B, T, n_heads, D)
    v_1 = attn1.v(r0).view(B, T, n_heads, D)

    cos1, sin1 = attn1.rotary(q1_1)
    q1_1 = apply_rotary_emb(q1_1, cos1, sin1)
    k1_1 = apply_rotary_emb(k1_1, cos1, sin1)
    q2_1 = apply_rotary_emb(q2_1, cos1, sin1)
    k2_1 = apply_rotary_emb(k2_1, cos1, sin1)

    scores1_1 = torch.einsum("bshd,bkhd->bhsk", q1_1, k1_1) / D
    scores2_1 = torch.einsum("bshd,bkhd->bhsk", q2_1, k2_1) / D
    pattern_1 = scores1_1 * scores2_1
    pattern_1 = pattern_1.masked_fill(~causal, 0.0)

    head_outputs_L1 = {}
    for h in range(n_heads):
        y_h = torch.einsum("bsk,bkd->bsd", pattern_1[:, h], v_1[:, :, h])
        out_h = y_h @ attn1.out.weight[:, h * D:(h + 1) * D].T
        head_outputs_L1[h] = out_h

    # Compute logit contributions via lm_head
    W_U = model.lm_head.weight  # (V, n_embd)

    direct_logits = E @ W_U.T  # (B, T, vocab)
    head_logit_contribs = {}
    for h in range(n_heads):
        head_logit_contribs[(0, h)] = head_outputs_L0[h] @ W_U.T
        head_logit_contribs[(1, h)] = head_outputs_L1[h] @ W_U.T

    # Verify: sum should equal full logits
    full_logits = direct_logits.clone()
    for key in head_logit_contribs:
        full_logits = full_logits + head_logit_contribs[key]

    # Compare to actual model
    actual_logits, _ = model(x)

    return direct_logits, head_logit_contribs, full_logits, actual_logits, pattern_0, pattern_1


def compute_logit_attribution(model, gen, n_batches=20, batch_size=64, seq_len=65):
    """Compute how much each head contributes to the correct-token logit, per rule."""
    rule_attribs = defaultdict(lambda: defaultdict(list))
    # rule_attribs[rule][component] = list of logit contributions

    n_heads = model.config.n_head
    components = ["direct"] + [f"L0H{h}" for h in range(n_heads)] + [f"L1H{h}" for h in range(n_heads)]

    recon_errors = []

    for batch_idx in range(n_batches):
        x, y, batch_labels, mask = make_batch(gen, batch_size, seq_len)
        with torch.no_grad():
            direct, head_contribs, full, actual, _, _ = forward_with_cache(model, x)

        # Check reconstruction
        err = (full - actual).abs().max().item()
        recon_errors.append(err)

        B, T, V = direct.shape

        for b in range(B):
            for t in range(T):
                lab = batch_labels[b][t] if t < len(batch_labels[b]) else "pad"
                if lab in ("pad", "pre_context"):
                    continue

                target_tok = y[b, t].item()

                # Direct path contribution to correct token logit
                rule_attribs[lab]["direct"].append(direct[b, t, target_tok].item())

                for h in range(n_heads):
                    rule_attribs[lab][f"L0H{h}"].append(
                        head_contribs[(0, h)][b, t, target_tok].item()
                    )
                    rule_attribs[lab][f"L1H{h}"].append(
                        head_contribs[(1, h)][b, t, target_tok].item()
                    )

    print(f"  Reconstruction max error: {max(recon_errors):.6f}")

    # Average per rule
    avg_attribs = {}
    for rule in rule_attribs:
        avg_attribs[rule] = {}
        for comp in components:
            vals = rule_attribs[rule][comp]
            avg_attribs[rule][comp] = sum(vals) / len(vals) if vals else 0.0

    return avg_attribs, components


def plot_logit_attribution(avg_attribs, components):
    """Stacked bar chart showing logit attribution per rule."""
    rules = sorted(avg_attribs.keys())
    n_rules = len(rules)
    n_comp = len(components)

    # Build matrix: (n_rules, n_comp)
    mat = np.zeros((n_rules, n_comp))
    for r_idx, rule in enumerate(rules):
        for c_idx, comp in enumerate(components):
            mat[r_idx, c_idx] = avg_attribs[rule].get(comp, 0.0)

    # Color map for components
    comp_colors = {
        "direct": "#333333",
        "L0H0": "#1f77b4", "L0H1": "#ff7f0e", "L0H2": "#2ca02c", "L0H3": "#d62728",
        "L1H0": "#9467bd", "L1H1": "#8c564b", "L1H2": "#e377c2", "L1H3": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_rules)
    width = 0.7

    # Separate positive and negative for stacked bars
    bottom_pos = np.zeros(n_rules)
    bottom_neg = np.zeros(n_rules)

    for c_idx, comp in enumerate(components):
        vals = mat[:, c_idx]
        pos_vals = np.maximum(vals, 0)
        neg_vals = np.minimum(vals, 0)

        color = comp_colors.get(comp, "#aaaaaa")
        ax.bar(x, pos_vals, width, bottom=bottom_pos, label=comp, color=color, alpha=0.85)
        ax.bar(x, neg_vals, width, bottom=bottom_neg, color=color, alpha=0.85)
        bottom_pos += pos_vals
        bottom_neg += neg_vals

    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Mean logit contribution to correct token")
    ax.set_title("Logit Attribution by Rule Type")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logit_attribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: logit_attribution.png")

    # Also make a normalized version (fraction of total)
    fig, ax = plt.subplots(figsize=(14, 6))
    # Normalize each rule to sum to 1 (absolute values)
    abs_mat = np.abs(mat)
    row_sums = abs_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_mat = abs_mat / row_sums

    bottom = np.zeros(n_rules)
    for c_idx, comp in enumerate(components):
        vals = norm_mat[:, c_idx]
        color = comp_colors.get(comp, "#aaaaaa")
        ax.bar(x, vals, width, bottom=bottom, label=comp, color=color, alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Fraction of |logit contribution|")
    ax.set_title("Logit Attribution (normalized) by Rule Type")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "logit_attribution_normalized.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("Saved: logit_attribution_normalized.png")


# %% 5. Attention pattern analysis on real data

def analyze_attention_patterns(model, gen, n_batches=10, batch_size=64, seq_len=65):
    """Compute mean attention distance per head, per rule."""
    n_heads = model.config.n_head

    # rule -> head -> list of (mean_distance, max_distance)
    rule_attn = defaultdict(lambda: defaultdict(list))

    for _ in range(n_batches):
        x, y, batch_labels, mask = make_batch(gen, batch_size, seq_len)
        with torch.no_grad():
            _, _, _, _, pattern_0, pattern_1 = forward_with_cache(model, x)

        B, T = x.shape[0], x.shape[1]
        # Distance matrix: d[i,j] = i - j (positive means j is earlier)
        dist = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)  # (T, T)
        dist = dist.float()

        for b in range(B):
            for t in range(T):
                lab = batch_labels[b][t] if t < len(batch_labels[b]) else "pad"
                if lab in ("pad", "pre_context"):
                    continue

                for h in range(n_heads):
                    # L0 attention pattern at position t
                    # Normalize: use abs values since bilinear patterns aren't softmaxed
                    attn_raw = pattern_0[b, h, t, :t + 1].abs()
                    total = attn_raw.sum().item()
                    if total > 1e-8:
                        attn_norm = attn_raw / total
                        mean_d = (attn_norm * dist[t, :t + 1]).sum().item()
                        max_pos = attn_raw.argmax().item()
                        rule_attn[lab][(0, h)].append((mean_d, t - max_pos))

                    attn_raw_1 = pattern_1[b, h, t, :t + 1].abs()
                    total_1 = attn_raw_1.sum().item()
                    if total_1 > 1e-8:
                        attn_norm_1 = attn_raw_1 / total_1
                        mean_d_1 = (attn_norm_1 * dist[t, :t + 1]).sum().item()
                        max_pos_1 = attn_raw_1.argmax().item()
                        rule_attn[lab][(1, h)].append((mean_d_1, t - max_pos_1))

    return rule_attn


def plot_attention_distances(rule_attn, n_heads):
    """Plot mean attention distance per head per rule."""
    rules = sorted(rule_attn.keys())
    head_keys = [(l, h) for l in range(2) for h in range(n_heads)]
    head_labels = [f"L{l}H{h}" for l, h in head_keys]

    # Matrix: (n_rules, n_heads_total) of mean attention distances
    mat = np.zeros((len(rules), len(head_keys)))
    for r_idx, rule in enumerate(rules):
        for h_idx, hk in enumerate(head_keys):
            vals = rule_attn[rule].get(hk, [])
            if vals:
                mat[r_idx, h_idx] = np.mean([v[0] for v in vals])

    fig, ax = plt.subplots(figsize=(10, max(6, len(rules) * 0.4)))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(head_labels)))
    ax.set_xticklabels(head_labels, fontsize=9)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules, fontsize=7)
    ax.set_xlabel("Head")
    ax.set_ylabel("Rule")
    ax.set_title("Mean attention distance by head and rule\n(larger = attends further back)")
    plt.colorbar(im, ax=ax, label="mean distance")

    # Annotate
    for i in range(len(rules)):
        for j in range(len(head_keys)):
            if abs(mat[i, j]) > 0.5:
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        fontsize=6, color="white" if mat[i, j] > mat.max() * 0.6 else "black")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attention_distances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: attention_distances.png")

    # Also: which head attends most for each rule?
    print("\n  Head with strongest attention per rule:")
    for r_idx, rule in enumerate(rules):
        row = mat[r_idx]
        best_head = head_labels[row.argmax()]
        print(f"    {rule:<25} -> {best_head} (mean_dist={row.max():.1f})")


# %% 6. Coverage analysis

def coverage_analysis(model, gen, avg_attribs, n_batches=10, batch_size=64, seq_len=65):
    """Measure per-rule loss and how much each head ablation hurts."""
    print("\n--- Coverage analysis ---")

    # Baseline per-rule loss
    all_rule_losses = defaultdict(list)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, labels_batch, mask = make_batch(gen, batch_size, seq_len)
            logits, _ = model(x)
            rl = per_rule_loss(logits, y, labels_batch, mask)
            for rule, loss in rl.items():
                all_rule_losses[rule].append(loss)

    baseline = {r: np.mean(v) for r, v in all_rule_losses.items()}

    # Head ablation: zero out each head's output weight, measure loss increase
    n_heads = model.config.n_head
    ablation_results = {}

    for layer in range(2):
        for head in range(n_heads):
            head_label = f"L{layer}H{head}"
            attn = model.blocks[layer].attn
            D = attn.head_dim

            # Save original
            orig = attn.out.weight[:, head * D:(head + 1) * D].data.clone()

            # Zero out
            attn.out.weight[:, head * D:(head + 1) * D].data.zero_()

            ablated_losses = defaultdict(list)
            with torch.no_grad():
                for _ in range(n_batches):
                    x, y, labels_batch, mask = make_batch(gen, batch_size, seq_len)
                    logits, _ = model(x)
                    rl = per_rule_loss(logits, y, labels_batch, mask)
                    for rule, loss in rl.items():
                        ablated_losses[rule].append(loss)

            ablated = {r: np.mean(v) for r, v in ablated_losses.items()}
            delta = {r: ablated.get(r, 0) - baseline.get(r, 0) for r in baseline}
            ablation_results[head_label] = delta

            # Restore
            attn.out.weight[:, head * D:(head + 1) * D].data.copy_(orig)

    # Plot ablation heatmap
    rules = sorted(baseline.keys())
    heads = [f"L{l}H{h}" for l in range(2) for h in range(n_heads)]
    mat = np.zeros((len(rules), len(heads)))
    for r_idx, rule in enumerate(rules):
        for h_idx, head in enumerate(heads):
            mat[r_idx, h_idx] = ablation_results[head].get(rule, 0)

    fig, ax = plt.subplots(figsize=(10, max(6, len(rules) * 0.4)))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels(heads, fontsize=9)
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules, fontsize=7)
    ax.set_title("Head Ablation: loss increase per rule\n(red = ablating this head hurts this rule)")
    ax.set_xlabel("Ablated head")
    ax.set_ylabel("Rule")
    plt.colorbar(im, ax=ax, label="loss increase")

    for i in range(len(rules)):
        for j in range(len(heads)):
            if abs(mat[i, j]) > 0.1:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(mat[i, j]) > np.abs(mat).max() * 0.5 else "black")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: ablation_heatmap.png")

    # Print summary
    print("\n  Baseline per-rule loss:")
    for rule in rules:
        print(f"    {rule:<25} {baseline[rule]:.4f}")

    print("\n  Most important head per rule (largest ablation increase):")
    for r_idx, rule in enumerate(rules):
        row = mat[r_idx]
        best_h = heads[row.argmax()]
        print(f"    {rule:<25} -> {best_h} (+{row.max():.3f})")

    # What's NOT explained? Rules where no single head ablation matters much
    print("\n  Rules with weak head dependence (max ablation delta < 0.2):")
    for r_idx, rule in enumerate(rules):
        max_delta = mat[r_idx].max()
        if max_delta < 0.2:
            print(f"    {rule:<25} max_delta={max_delta:.3f} (mostly direct path?)")

    return baseline, ablation_results


# %% 7. Direct path analysis

def plot_direct_path(model, labels, categories, sorted_indices, sorted_labels, boundaries):
    """Labeled direct path with top predictions per token."""
    E = model.embed.weight
    U = model.lm_head.weight

    with torch.no_grad():
        D = U @ E.T  # (V, V): for each input token, direct logit for each output token
        D_sorted = D[sorted_indices][:, sorted_indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Full matrix
    labeled_heatmap(axes[0], D_sorted, sorted_labels, sorted_labels,
                    boundaries, "Direct path: lm_head @ embed^T")
    axes[0].set_ylabel("output token")
    axes[0].set_xlabel("input token")

    # Top-3 predictions per token (text table)
    ax = axes[1]
    ax.axis("off")
    rows = []
    for i in range(len(labels)):
        logits = D[:, i]  # output logits given input token i
        top3 = logits.topk(3)
        preds = ", ".join(f"{labels[j]}({v:.2f})" for j, v in zip(top3.indices, top3.values))
        cat = categories.get(i, "?")
        rows.append(f"{labels[i]:>10} ({cat:>6}) -> {preds}")

    text = "\n".join(rows)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=6,
            verticalalignment="top", fontfamily="monospace")
    ax.set_title("Direct path: top-3 predictions per input token", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "direct_path_labeled.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: direct_path_labeled.png")


# %% Main

def main():
    print("Loading model...")
    model, config = load_model()
    labels, categories, sorted_indices, sorted_labels, boundaries = get_category_order()
    gen = LanguageV3(seed=42)
    print(f"Model: {config.n_layer}L {config.n_head}H d={config.n_embd}, vocab={config.vocab_size}")

    # 1. Per-head evidence figures with labels
    print("\n=== 1. Per-head QK/OV evidence ===")
    for layer in range(config.n_layer):
        for head in range(config.n_head):
            plot_head_evidence(model, layer, head, labels, categories,
                               sorted_indices, sorted_labels, boundaries,
                               distances=[1, 3, 8])

    # 2. Tucker factor bar charts at peak distance
    print("\n=== 2. Tucker factor bar charts ===")
    peak_distances = {
        (0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1,
        (1, 0): 2, (1, 1): 1, (1, 2): 1, (1, 3): 5,
    }
    for (layer, head), d in peak_distances.items():
        plot_tucker_factors(model, layer, head, d, labels, categories)

    # 3. Direct path
    print("\n=== 3. Direct path ===")
    plot_direct_path(model, labels, categories, sorted_indices, sorted_labels, boundaries)

    # 4. Composition scores
    print("\n=== 4. L0 -> L1 composition ===")
    v_comp, k_comp, q_comp = plot_composition(model)
    print("  V-composition (which L0 heads feed into which L1 heads):")
    for h0 in range(config.n_head):
        for h1 in range(config.n_head):
            if v_comp[h0, h1] > 0.15:
                print(f"    L0H{h0} -> L1H{h1}: V-comp={v_comp[h0, h1]:.3f}, "
                      f"K-comp={k_comp[h0, h1]:.3f}, Q-comp={q_comp[h0, h1]:.3f}")

    # 5. Logit attribution
    print("\n=== 5. Logit attribution per rule ===")
    avg_attribs, components = compute_logit_attribution(model, gen)
    plot_logit_attribution(avg_attribs, components)

    # Print summary table
    rules = sorted(avg_attribs.keys())
    print(f"\n  {'Rule':<25} {'Direct':>8} {'L0H0':>8} {'L0H1':>8} {'L0H2':>8} {'L0H3':>8}"
          f" {'L1H0':>8} {'L1H1':>8} {'L1H2':>8} {'L1H3':>8}")
    print("  " + "-" * 100)
    for rule in rules:
        vals = [avg_attribs[rule].get(c, 0) for c in components]
        line = f"  {rule:<25}"
        for v in vals:
            line += f" {v:>8.3f}"
        line += f"  sum={sum(vals):.3f}"
        print(line)

    # 6. Attention patterns on real data
    print("\n=== 6. Attention patterns ===")
    rule_attn = analyze_attention_patterns(model, gen)
    plot_attention_distances(rule_attn, config.n_head)

    # 7. Head ablation / coverage
    print("\n=== 7. Head ablation ===")
    baseline, ablation_results = coverage_analysis(model, gen, avg_attribs)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

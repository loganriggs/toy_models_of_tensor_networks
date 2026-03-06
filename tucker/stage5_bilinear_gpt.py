"""Stage 5: Tucker decomposition of BilinearGPT attention heads.

For each head, construct the full 3rd-order circuit tensor T[q,k,o] in token space:
  T[q,k,o] = QK1[q,k] * QK2[q,k] * OV[k,o]

where:
  QK1[q,k] = E[q] @ W_Q1_h @ W_K1_h^T @ E[k]^T / D
  QK2[q,k] = E[q] @ W_Q2_h @ W_K2_h^T @ E[k]^T / D
  OV[k,o]  = E[k] @ W_V_h @ W_O_h @ lm_head[o]^T

Then apply Tucker + rotation to find interpretable circuit structure.

Run: python tucker/stage5_bilinear_gpt.py
"""

# %% Imports

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "attn_circuits"))

from model import BilinearGPT, Config
from tucker_pipeline import tucker_pipeline, hosvd, off_diagonal_mass, rotate_core

FIGURES_DIR = Path(__file__).parent / "figures" / "stage5"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# %% Load model

def load_model():
    config_path = ROOT / "configs" / "bilinear_gpt_v3_all_2L_4H_d64_5000steps.json"
    model_path = ROOT / "models" / "bilinear_gpt_v3_all_2L_4H_d64_5000steps.pt"

    with open(config_path) as f:
        cfg = json.load(f)
    config = Config(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        seq_len=cfg["seq_len"],
        block_has_mlp=tuple(cfg["block_has_mlp"]),
    )
    model = BilinearGPT(config)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


# %% Extract per-head circuit tensors

def extract_head_weights(model, layer_idx, head_idx):
    """Extract Q1, K1, Q2, K2, V, O weight slices for a specific head."""
    block = model.blocks[layer_idx]
    attn = block.attn
    d = attn.head_dim
    h = head_idx

    Q1 = attn.q1.weight[h * d : (h + 1) * d, :]  # (d, n_embd)
    K1 = attn.k1.weight[h * d : (h + 1) * d, :]
    Q2 = attn.q2.weight[h * d : (h + 1) * d, :]
    K2 = attn.k2.weight[h * d : (h + 1) * d, :]
    V = attn.v.weight[h * d : (h + 1) * d, :]
    O = attn.out.weight[:, h * d : (h + 1) * d]  # (n_embd, d)

    return Q1, K1, Q2, K2, V, O


def build_circuit_tensor(model, layer_idx, head_idx):
    """Build the 3rd-order circuit tensor T[q,k,o] for a head.

    T[q,k,o] = QK1[q,k] * QK2[q,k] * OV[k,o]

    where QK and OV are embedded in token space (content-only, no RoPE).
    Returns tensor of shape (V, V, V) where V = vocab_size.
    """
    E = model.embed.weight  # (V, n_embd)
    U = model.lm_head.weight  # (V, n_embd)
    Q1, K1, Q2, K2, V, O = extract_head_weights(model, layer_idx, head_idx)
    D = Q1.shape[0]  # head_dim

    # QK circuits in token space (V x V)
    QK1 = (E @ Q1.T @ K1 @ E.T) / D  # (V, V)
    QK2 = (E @ Q2.T @ K2 @ E.T) / D

    # OV circuit in token space: E @ V^T @ O^T @ U^T = (V, V)
    OV = E @ V.T @ O.T @ U.T  # (V, V): key_token -> output_token logit

    # Full circuit tensor: T[q,k,o] = QK1[q,k] * QK2[q,k] * OV[k,o]
    QK = QK1 * QK2  # (V, V) — squared bilinear attention pattern
    T = torch.einsum("qk,ko->qko", QK, OV)  # (V, V, V)

    return T, QK1, QK2, QK, OV


def build_direct_path(model):
    """Direct path: lm_head @ embed^T (V x V)."""
    return model.lm_head.weight @ model.embed.weight.T  # (V, V)


# %% Token labels from v3 generator

def get_token_labels():
    """Return token names and category info for v3 language."""
    from generator_v3 import LanguageV3
    gen = LanguageV3(seed=42)
    labels = gen.vocab  # list of token names
    categories = {}
    for tok_id, name in enumerate(labels):
        cat = gen.token2cat.get(name, "STRUCT")
        categories[tok_id] = cat
    return labels, categories


# %% Run Tucker on each head

def analyze_head(model, layer_idx, head_idx, labels, categories, top_k=5):
    """Run Tucker analysis on a single attention head's circuit tensor."""
    with torch.no_grad():
        T, QK1, QK2, QK, OV = build_circuit_tensor(model, layer_idx, head_idx)

    V = T.shape[0]
    head_label = f"L{layer_idx}H{head_idx}"
    print(f"\n{'='*60}")
    print(f"Head {head_label}")
    print(f"{'='*60}")
    print(f"  Circuit tensor shape: {T.shape}, norm: {T.norm():.4f}")
    print(f"  QK pattern norm: {QK.norm():.4f}")
    print(f"  OV matrix norm: {OV.norm():.4f}")

    # Rank search (fast mode)
    print(f"\n  --- Rank search ---")
    max_rank = min(6, V)
    recon_errors = []
    odm_values = []
    for r in range(1, max_rank + 1):
        try:
            _, _, _, _, metrics = tucker_pipeline(T, r, fast=True)
            recon_errors.append(metrics["recon_error"])
            odm_values.append(metrics["odm_after"])
            print(f"  rank={r}: recon={metrics['recon_error']:.4f}, odm={metrics['odm_after']:.4f}",
                  flush=True)
        except Exception as e:
            print(f"  rank={r}: FAILED ({e})")
            recon_errors.append(1.0)
            odm_values.append(1.0)

    # Find elbow (largest drop in recon error)
    drops = [recon_errors[i] - recon_errors[i + 1] for i in range(len(recon_errors) - 1)]
    best_rank = drops.index(max(drops)) + 2 if drops else 1
    # Cap and ensure reasonable recon
    for r in range(1, len(recon_errors) + 1):
        if recon_errors[r - 1] < 0.15:
            best_rank = r
            break
    best_rank = min(best_rank, 5)

    print(f"\n  Selected rank: {best_rank}")
    G_rot, A_rot, B_rot, C_rot, metrics = tucker_pipeline(T, best_rank)  # full quality

    # Interpret factor columns in token space
    print(f"\n  --- Query factors (A_rot) ---")
    for c in range(best_rank):
        col = A_rot[:, c]
        top_pos = col.abs().topk(top_k)
        top_neg = col.topk(top_k, largest=False)
        pos_tokens = [(labels[i], col[i].item()) for i in top_pos.indices]
        print(f"    Component {c} (top tokens): {', '.join(f'{t}({v:.3f})' for t, v in pos_tokens)}")

    print(f"\n  --- Key factors (B_rot) ---")
    for c in range(best_rank):
        col = B_rot[:, c]
        top_pos = col.abs().topk(top_k)
        pos_tokens = [(labels[i], col[i].item()) for i in top_pos.indices]
        print(f"    Component {c} (top tokens): {', '.join(f'{t}({v:.3f})' for t, v in pos_tokens)}")

    print(f"\n  --- Output factors (C_rot) ---")
    for c in range(best_rank):
        col = C_rot[:, c]
        top_pos = col.abs().topk(top_k)
        pos_tokens = [(labels[i], col[i].item()) for i in top_pos.indices]
        print(f"    Component {c} (top tokens): {', '.join(f'{t}({v:.3f})' for t, v in pos_tokens)}")

    # Core diagonal
    r = min(best_rank, *G_rot.shape)
    diag = [G_rot[i, i, i].item() for i in range(r)]
    print(f"\n  Core diagonal: {[f'{d:.4f}' for d in diag]}")
    print(f"  Off-diagonal mass: {metrics['odm_after']:.4f}")

    # Interpret circuits: for each diagonal entry, describe the rule
    print(f"\n  --- Recovered circuits ---")
    for c in range(r):
        g_val = G_rot[c, c, c].item()
        if abs(g_val) < 0.01 * abs(max(diag, key=abs)):
            continue
        q_top = labels[A_rot[:, c].abs().argmax()]
        k_top = labels[B_rot[:, c].abs().argmax()]
        o_top = labels[C_rot[:, c].abs().argmax()]
        q_cat = categories.get(A_rot[:, c].abs().argmax().item(), "?")
        k_cat = categories.get(B_rot[:, c].abs().argmax().item(), "?")
        o_cat = categories.get(C_rot[:, c].abs().argmax().item(), "?")
        print(f"    Circuit {c}: query={q_top}({q_cat}) x key={k_top}({k_cat}) -> {o_top}({o_cat}) [G={g_val:.4f}]")

    return {
        "head": head_label,
        "best_rank": best_rank,
        "recon_errors": recon_errors,
        "odm_values": odm_values,
        "metrics": metrics,
        "G_rot": G_rot,
        "A_rot": A_rot,
        "B_rot": B_rot,
        "C_rot": C_rot,
    }


# %% Visualize

def plot_rank_search_all(results):
    """Plot rank search curves for all heads."""
    n_heads = len(results)
    fig, axes = plt.subplots(2, n_heads, figsize=(4 * n_heads, 8))

    for idx, res in enumerate(results):
        ax1 = axes[0, idx]
        ax1.plot(range(1, len(res["recon_errors"]) + 1), res["recon_errors"], "o-")
        ax1.axvline(res["best_rank"], color="r", linestyle="--", alpha=0.5)
        ax1.set_title(f'{res["head"]} recon')
        ax1.set_xlabel("rank")
        ax1.set_ylabel("recon error")

        ax2 = axes[1, idx]
        ax2.plot(range(1, len(res["odm_values"]) + 1), res["odm_values"], "o-")
        ax2.axvline(res["best_rank"], color="r", linestyle="--", alpha=0.5)
        ax2.set_title(f'{res["head"]} ODM')
        ax2.set_xlabel("rank")
        ax2.set_ylabel("off-diag mass")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rank_search_all_heads.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'rank_search_all_heads.png'}")


def plot_qk_ov_matrices(model, labels, categories):
    """Plot QK and OV matrices for all heads."""
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    total_heads = n_layers * n_heads

    # Category ordering for nicer visualization
    cat_order = ["NOUN", "PLACE", "VERB_T", "VERB_I", "ADJ", "FUNC", "PUNCT", "OPEN", "CLOSE", "STRUCT"]
    sorted_indices = sorted(range(len(labels)), key=lambda i: (cat_order.index(categories.get(i, "STRUCT")), labels[i]))
    sorted_labels = [labels[i] for i in sorted_indices]

    # QK patterns
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for l in range(n_layers):
            for h in range(n_heads):
                _, QK1, QK2, QK, OV = build_circuit_tensor(model, l, h)
                QK_sorted = QK[sorted_indices][:, sorted_indices]
                ax = axes[l, h]
                im = ax.imshow(QK_sorted.numpy(), cmap="RdBu_r", aspect="auto",
                               vmin=-QK_sorted.abs().max().item(), vmax=QK_sorted.abs().max().item())
                ax.set_title(f"L{l}H{h} QK pattern")
                plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "qk_patterns.png", dpi=150)
    plt.close()

    # OV matrices
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for l in range(n_layers):
            for h in range(n_heads):
                _, _, _, _, OV = build_circuit_tensor(model, l, h)
                OV_sorted = OV[sorted_indices][:, sorted_indices]
                ax = axes[l, h]
                im = ax.imshow(OV_sorted.numpy(), cmap="RdBu_r", aspect="auto",
                               vmin=-OV_sorted.abs().max().item(), vmax=OV_sorted.abs().max().item())
                ax.set_title(f"L{l}H{h} OV circuit")
                plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ov_circuits.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'qk_patterns.png'}")
    print(f"Saved: {FIGURES_DIR / 'ov_circuits.png'}")


def plot_direct_path(model, labels, categories):
    """Plot the direct path (embed -> lm_head)."""
    cat_order = ["NOUN", "PLACE", "VERB_T", "VERB_I", "ADJ", "FUNC", "PUNCT", "OPEN", "CLOSE", "STRUCT"]
    sorted_indices = sorted(range(len(labels)), key=lambda i: (cat_order.index(categories.get(i, "STRUCT")), labels[i]))

    with torch.no_grad():
        D = build_direct_path(model)
        D_sorted = D[sorted_indices][:, sorted_indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(D_sorted.numpy(), cmap="RdBu_r", aspect="auto",
                   vmin=-D_sorted.abs().max().item(), vmax=D_sorted.abs().max().item())
    ax.set_title("Direct path: lm_head @ embed^T")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "direct_path.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'direct_path.png'}")


# %% Cross-reference with known rules

def cross_reference_rules(results, model, labels, categories):
    """Compare Tucker-recovered circuits with known v3 language rules."""
    from generator_v3 import LanguageV3
    gen = LanguageV3(seed=42)

    print("\n" + "=" * 60)
    print("CROSS-REFERENCE WITH KNOWN RULES")
    print("=" * 60)

    # Known rule types and what they look like in circuit form:
    # - place_bigram: specific token -> specific token (e.g., park -> the)
    # - noun_bigram: specific noun -> weighted verb distribution
    # - cat_bigram: category -> category (e.g., NOUN -> VERB_T with high prob)
    # - tok_trigram: (trigger1, trigger2) -> specific output
    # - skip_bigram: anchor...trigger -> specific output
    # - induction: previous-token copying

    # For each head, check if the top Tucker circuits match known rules
    for res in results:
        head = res["head"]
        G = res["G_rot"]
        A = res["A_rot"]
        B = res["B_rot"]
        C = res["C_rot"]
        r = min(G.shape)

        print(f"\n--- {head} ---")
        for c in range(r):
            g_val = G[c, c, c].item()
            if abs(g_val) < 0.001:
                continue

            # Top query/key/output tokens
            q_col = A[:, c]
            k_col = B[:, c]
            o_col = C[:, c]

            q_top5 = q_col.abs().topk(5).indices.tolist()
            k_top5 = k_col.abs().topk(5).indices.tolist()
            o_top5 = o_col.abs().topk(5).indices.tolist()

            q_cats = [categories.get(i, "?") for i in q_top5]
            k_cats = [categories.get(i, "?") for i in k_top5]
            o_cats = [categories.get(i, "?") for i in o_top5]

            # Check if this is a category-level circuit
            q_dominant_cat = max(set(q_cats), key=q_cats.count) if q_cats else "?"
            k_dominant_cat = max(set(k_cats), key=k_cats.count) if k_cats else "?"
            o_dominant_cat = max(set(o_cats), key=o_cats.count) if o_cats else "?"

            rule_guess = "unknown"
            if q_dominant_cat == k_dominant_cat == o_dominant_cat:
                rule_guess = f"self-loop ({q_dominant_cat})"
            elif len(set(q_cats)) == 1 and len(set(k_cats)) == 1:
                rule_guess = f"cat_bigram? ({q_dominant_cat} x {k_dominant_cat} -> {o_dominant_cat})"
            elif len(set([labels[i] for i in q_top5[:2]])) <= 2:
                rule_guess = f"tok_specific ({labels[q_top5[0]]} x {labels[k_top5[0]]} -> {labels[o_top5[0]]})"

            print(f"  Circuit {c} (G={g_val:.4f}): "
                  f"Q={q_dominant_cat}, K={k_dominant_cat} -> O={o_dominant_cat} "
                  f"[{rule_guess}]")


# %% Main

def main():
    print("Loading model...")
    model, config = load_model()
    labels, categories = get_token_labels()
    print(f"Model: {config.n_layer}L {config.n_head}H d={config.n_embd}, vocab={config.vocab_size}")

    # Plot basic matrices first
    print("\nPlotting direct path, QK patterns, OV circuits...")
    plot_direct_path(model, labels, categories)
    plot_qk_ov_matrices(model, labels, categories)

    # Tucker analysis of each head
    results = []
    for layer in range(config.n_layer):
        for head in range(config.n_head):
            res = analyze_head(model, layer, head, labels, categories)
            results.append(res)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Head':<8} {'Rank':>5} {'Recon':>8} {'ODM':>8} {'||T||':>8}")
    for res in results:
        r = res["best_rank"]
        recon = res["metrics"]["recon_error"]
        odm = res["metrics"]["odm_after"]
        print(f"{res['head']:<8} {r:>5} {recon:>8.4f} {odm:>8.4f}")

    plot_rank_search_all(results)
    cross_reference_rules(results, model, labels, categories)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

"""Stage 5b: Position-conditioned Tucker decomposition of BilinearGPT.

The content-only circuit tensor T[q,k,o] from stage5 ignores RoPE positional
information. Here we build a separate tensor T_d[q,k,o] for each relative
distance d, incorporating the RoPE rotation matrix R(d).

For relative distance d (key is d positions before query):
  QK1_d[q,k] = E[q] @ Q1 @ R(d) @ K1^T @ E[k]^T / D
  QK2_d[q,k] = E[q] @ Q2 @ R(d) @ K2^T @ E[k]^T / D
  T_d[q,k,o] = QK1_d[q,k] * QK2_d[q,k] * OV[k,o]

where R(d) is the RoPE relative rotation matrix (block-diagonal with 2x2 blocks).
OV doesn't depend on position since it only involves V and O weights.

No combinatorial blowup: just iterate d=1,2,...,max_dist.

Run: python tucker/stage5b_position_conditioned.py
"""

# %% Imports

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "attn_circuits"))

from model import BilinearGPT, Config
from tucker_pipeline import tucker_pipeline, hosvd, off_diagonal_mass

FIGURES_DIR = Path(__file__).parent / "figures" / "stage5b"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# %% Load model (reuse from stage5)

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


def get_token_labels():
    from generator_v3 import LanguageV3
    gen = LanguageV3(seed=42)
    labels = gen.vocab
    categories = {}
    for tok_id, name in enumerate(labels):
        cat = gen.token2cat.get(name, "STRUCT")
        categories[tok_id] = cat
    return labels, categories


# %% RoPE rotation matrix for relative distance d

def rope_rotation_matrix(head_dim, d, base=10000):
    """Build the D x D rotation matrix R(d) for relative distance d.

    RoPE applies per-pair rotations. For dimension pair (i, i+D/2), the
    2x2 block is [[cos(d*f_i), sin(d*f_i)], [-sin(d*f_i), cos(d*f_i)]].

    The key identity: q_rot(p) @ k_rot(p-d)^T = q @ R(d) @ k^T
    where R(d) is block-diagonal with these 2x2 rotation blocks.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # inv_freq has shape (half,)

    R = torch.zeros(head_dim, head_dim)
    for i in range(half):
        angle = d * inv_freq[i].item()
        c, s = np.cos(angle), np.sin(angle)
        # The RoPE encoding splits x into [x1, x2] where x1 = x[:half], x2 = x[half:]
        # x_rot = [x1*cos + x2*sin, -x1*sin + x2*cos]
        # So for q at position p and k at position p-d:
        # q_rot(p) @ k_rot(p-d)^T has the relative rotation R(d)
        # R(d) block for dimension pair (i, i+half):
        #   [[cos(d*f), sin(d*f)],
        #    [-sin(d*f), cos(d*f)]]
        # acting on the (i, i+half) subspace
        R[i, i] = c
        R[i, i + half] = -s
        R[i + half, i] = s
        R[i + half, i + half] = c

    return R


# %% Extract head weights (same as stage5)

def extract_head_weights(model, layer_idx, head_idx):
    block = model.blocks[layer_idx]
    attn = block.attn
    d = attn.head_dim
    h = head_idx
    Q1 = attn.q1.weight[h * d : (h + 1) * d, :]
    K1 = attn.k1.weight[h * d : (h + 1) * d, :]
    Q2 = attn.q2.weight[h * d : (h + 1) * d, :]
    K2 = attn.k2.weight[h * d : (h + 1) * d, :]
    V = attn.v.weight[h * d : (h + 1) * d, :]
    O = attn.out.weight[:, h * d : (h + 1) * d]
    return Q1, K1, Q2, K2, V, O


# %% Build position-conditioned circuit tensor

def build_circuit_tensor_at_distance(model, layer_idx, head_idx, d):
    """Build T_d[q,k,o] for relative distance d.

    QK1_d = E @ Q1 @ R(d) @ K1^T @ E^T / D
    QK2_d = E @ Q2 @ R(d) @ K2^T @ E^T / D
    T_d = (QK1_d * QK2_d) x OV
    """
    E = model.embed.weight  # (V, n_embd)
    U = model.lm_head.weight  # (V, n_embd)
    Q1, K1, Q2, K2, Vw, O = extract_head_weights(model, layer_idx, head_idx)
    D = Q1.shape[0]  # head_dim

    R_d = rope_rotation_matrix(D, d).to(E.device)

    # QK with position: E @ Q @ R(d) @ K^T @ E^T / D
    QK1_d = (E @ Q1.T @ R_d @ K1 @ E.T) / D
    QK2_d = (E @ Q2.T @ R_d @ K2 @ E.T) / D
    QK_d = QK1_d * QK2_d

    # OV is position-independent
    OV = E @ Vw.T @ O.T @ U.T

    T_d = torch.einsum("qk,ko->qko", QK_d, OV)
    return T_d, QK1_d, QK2_d, QK_d, OV


def build_content_only_tensor(model, layer_idx, head_idx):
    """Content-only tensor (d=0, i.e., R=I). Same as stage5."""
    return build_circuit_tensor_at_distance(model, layer_idx, head_idx, d=0)


# %% Analyze how circuit structure changes with distance

def analyze_distance_profile(model, layer_idx, head_idx, labels, categories,
                             max_dist=15):
    """For one head, compute ||T_d|| and Tucker rank profile across distances."""
    head_label = f"L{layer_idx}H{head_idx}"
    print(f"\n{'='*60}")
    print(f"Head {head_label}: Distance profile")
    print(f"{'='*60}")

    norms = []
    qk_norms = []
    top_circuits = []

    with torch.no_grad():
        # Content-only baseline
        T0, _, _, QK0, OV = build_content_only_tensor(model, layer_idx, head_idx)
        ov_norm = OV.norm().item()
        print(f"  OV norm (position-independent): {ov_norm:.4f}")

        for d in range(1, max_dist + 1):
            T_d, _, _, QK_d, _ = build_circuit_tensor_at_distance(
                model, layer_idx, head_idx, d
            )
            t_norm = T_d.norm().item()
            qk_norm = QK_d.norm().item()
            norms.append(t_norm)
            qk_norms.append(qk_norm)

            # Top circuit: argmax of |T_d[q,k,o]|
            flat_idx = T_d.abs().argmax().item()
            V = T_d.shape[0]
            q_idx = flat_idx // (V * V)
            k_idx = (flat_idx % (V * V)) // V
            o_idx = flat_idx % V
            val = T_d[q_idx, k_idx, o_idx].item()
            top_circuits.append((d, labels[q_idx], labels[k_idx], labels[o_idx], val))

            print(f"  d={d:>2}: ||T_d||={t_norm:>8.3f}  ||QK_d||={qk_norm:>8.3f}"
                  f"  top: {labels[q_idx]}×{labels[k_idx]}→{labels[o_idx]} ({val:+.3f})")

    return {
        "head": head_label,
        "norms": norms,
        "qk_norms": qk_norms,
        "top_circuits": top_circuits,
        "ov_norm": ov_norm,
    }


# %% Tucker at specific distances

def tucker_at_distance(model, layer_idx, head_idx, d, labels, categories, rank=None):
    """Run Tucker decomposition on T_d for a specific distance."""
    head_label = f"L{layer_idx}H{head_idx}"

    with torch.no_grad():
        T_d, QK1_d, QK2_d, QK_d, OV = build_circuit_tensor_at_distance(
            model, layer_idx, head_idx, d
        )

    if T_d.norm() < 1e-6:
        print(f"  {head_label} d={d}: tensor near-zero, skipping")
        return None

    # Auto rank selection: try 1-4, pick where recon < 0.15
    if rank is None:
        for r in range(1, 5):
            try:
                _, _, _, _, m = tucker_pipeline(T_d, r, fast=True)
                if m["recon_error"] < 0.15:
                    rank = r
                    break
            except Exception:
                continue
        if rank is None:
            rank = 4

    G_rot, A_rot, B_rot, C_rot, metrics = tucker_pipeline(T_d, rank, fast=True)

    r = min(rank, *G_rot.shape)
    diag = [G_rot[i, i, i].item() for i in range(r)]

    print(f"\n  {head_label} d={d}: rank={rank}, recon={metrics['recon_error']:.4f}, "
          f"odm={metrics['odm_after']:.4f}")
    print(f"    Core diagonal: {[f'{d:.3f}' for d in diag]}")

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
        print(f"    Circuit {c}: {q_top}({q_cat}) × {k_top}({k_cat}) → {o_top}({o_cat}) [G={g_val:.3f}]")

    return {
        "head": head_label, "d": d, "rank": rank,
        "G_rot": G_rot, "A_rot": A_rot, "B_rot": B_rot, "C_rot": C_rot,
        "metrics": metrics, "QK_d": QK_d,
    }


# %% Visualization

def plot_distance_profiles(profiles, n_layers, n_heads):
    """Plot ||T_d|| vs d for all heads."""
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers),
                             squeeze=False)

    for prof in profiles:
        head = prof["head"]
        layer = int(head[1])
        h = int(head[3])
        ax = axes[layer, h]
        dists = list(range(1, len(prof["norms"]) + 1))
        ax.plot(dists, prof["norms"], "o-", label="||T_d||", markersize=4)
        ax.plot(dists, prof["qk_norms"], "s--", label="||QK_d||", markersize=3, alpha=0.6)
        ax.set_title(head)
        ax.set_xlabel("relative distance d")
        ax.set_ylabel("norm")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distance_profiles.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'distance_profiles.png'}")


def plot_qk_at_distances(model, layer_idx, head_idx, labels, categories,
                         distances=[1, 2, 3, 5, 10]):
    """Plot QK_d matrices at selected distances for one head."""
    head_label = f"L{layer_idx}H{head_idx}"

    cat_order = ["NOUN", "PLACE", "VERB_T", "VERB_I", "ADJ", "FUNC",
                 "PUNCT", "OPEN", "CLOSE", "STRUCT"]
    sorted_indices = sorted(
        range(len(labels)),
        key=lambda i: (cat_order.index(categories.get(i, "STRUCT")), labels[i])
    )

    n = len(distances)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    with torch.no_grad():
        for idx, d in enumerate(distances):
            _, _, _, QK_d, _ = build_circuit_tensor_at_distance(
                model, layer_idx, head_idx, d
            )
            QK_sorted = QK_d[sorted_indices][:, sorted_indices]
            ax = axes[idx]
            vmax = QK_sorted.abs().max().item()
            if vmax < 1e-8:
                vmax = 1.0
            im = ax.imshow(QK_sorted.numpy(), cmap="RdBu_r", aspect="auto",
                           vmin=-vmax, vmax=vmax)
            ax.set_title(f"{head_label} QK d={d}")
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"qk_distances_{head_label}.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / f'qk_distances_{head_label}.png'}")


def plot_tucker_summary(tucker_results, labels, categories):
    """Summary heatmap: for each (head, distance), show top circuit description."""
    # Group by head
    heads = sorted(set(r["head"] for r in tucker_results if r is not None))
    distances = sorted(set(r["d"] for r in tucker_results if r is not None))

    fig, axes = plt.subplots(len(heads), 1, figsize=(12, 3 * len(heads)))
    if len(heads) == 1:
        axes = [axes]

    for h_idx, head in enumerate(heads):
        ax = axes[h_idx]
        head_results = [r for r in tucker_results if r is not None and r["head"] == head]
        head_results.sort(key=lambda r: r["d"])

        # Build matrix: rows = Tucker components, cols = distances
        max_rank = max(r["rank"] for r in head_results)
        mat = np.zeros((max_rank, len(head_results)))
        x_labels = []
        annotations = []

        for col, r in enumerate(head_results):
            x_labels.append(f"d={r['d']}")
            G = r["G_rot"]
            rank = min(r["rank"], *G.shape)
            col_annots = []
            for c in range(rank):
                mat[c, col] = abs(G[c, c, c].item())
                q = labels[r["A_rot"][:, c].abs().argmax()]
                k = labels[r["B_rot"][:, c].abs().argmax()]
                o = labels[r["C_rot"][:, c].abs().argmax()]
                col_annots.append(f"{q}×{k}→{o}")
            annotations.append(col_annots)

        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_yticks(range(max_rank))
        ax.set_yticklabels([f"comp {c}" for c in range(max_rank)], fontsize=8)
        ax.set_title(f"{head}: Tucker circuits by distance")
        plt.colorbar(im, ax=ax, label="|G[i,i,i]|")

        # Annotate cells
        for col in range(len(annotations)):
            for row, txt in enumerate(annotations[col]):
                if mat[row, col] > 0.01:
                    ax.text(col, row, txt, ha="center", va="center", fontsize=5,
                            color="white" if mat[row, col] > mat.max() * 0.6 else "black")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tucker_by_distance.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'tucker_by_distance.png'}")


# %% Verify RoPE rotation matrix

def verify_rope_matrix(model):
    """Verify that R(d) correctly reproduces the RoPE relative attention scores."""
    block = model.blocks[0]
    attn = block.attn
    D = attn.head_dim

    # Make random q, k vectors
    torch.manual_seed(123)
    q = torch.randn(1, 2, 1, D)  # batch=1, seq=2, head=1, dim=D
    k = torch.randn(1, 2, 1, D)

    # Method 1: Apply RoPE at positions 5 and 3 (d=2), then dot product
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    from model import apply_rotary_emb
    for pos in [3, 5]:
        freqs = torch.outer(torch.tensor([float(pos)]), inv_freq)
        cos_p = freqs.cos().unsqueeze(0).unsqueeze(2)
        sin_p = freqs.sin().unsqueeze(0).unsqueeze(2)
        if pos == 5:
            q_rot = apply_rotary_emb(q[:, :1], cos_p, sin_p)
        else:
            k_rot = apply_rotary_emb(k[:, :1], cos_p, sin_p)

    score_rope = (q_rot * k_rot).sum().item()

    # Method 2: q @ R(d=2) @ k^T
    R2 = rope_rotation_matrix(D, d=2)
    score_matrix = (q[0, 0, 0] @ R2 @ k[0, 0, 0]).item()

    print(f"  RoPE verification: rope_score={score_rope:.6f}, matrix_score={score_matrix:.6f}")
    assert abs(score_rope - score_matrix) < 1e-4, f"Mismatch: {abs(score_rope - score_matrix)}"
    print("  PASSED")


# %% Main

def main():
    print("Loading model...")
    model, config = load_model()
    labels, categories = get_token_labels()
    print(f"Model: {config.n_layer}L {config.n_head}H d={config.n_embd}, vocab={config.vocab_size}")

    # Verify RoPE matrix correctness
    print("\nVerifying RoPE rotation matrix...")
    verify_rope_matrix(model)

    # Distance profiles for all heads
    print("\n--- Distance profiles ---")
    profiles = []
    with torch.no_grad():
        for layer in range(config.n_layer):
            for head in range(config.n_head):
                prof = analyze_distance_profile(model, layer, head, labels, categories,
                                                max_dist=15)
                profiles.append(prof)

    plot_distance_profiles(profiles, config.n_layer, config.n_head)

    # Identify interesting heads (high norm at specific distances)
    print("\n--- Identifying position-sensitive heads ---")
    for prof in profiles:
        norms = prof["norms"]
        peak_d = norms.index(max(norms)) + 1
        ratio = max(norms) / (sum(norms) / len(norms) + 1e-8)
        if ratio > 2.0:
            print(f"  {prof['head']}: peaks at d={peak_d} (ratio={ratio:.1f}x avg)")
        else:
            print(f"  {prof['head']}: relatively uniform (max/avg={ratio:.1f})")

    # Tucker decomposition at key distances for each head
    print("\n--- Tucker at key distances ---")
    key_distances = [1, 2, 3, 5, 8, 12]
    tucker_results = []
    with torch.no_grad():
        for layer in range(config.n_layer):
            for head in range(config.n_head):
                # QK pattern visualization
                plot_qk_at_distances(model, layer, head, labels, categories,
                                     distances=[1, 2, 5, 10])
                for d in key_distances:
                    res = tucker_at_distance(model, layer, head, d, labels, categories)
                    if res is not None:
                        tucker_results.append(res)

    if tucker_results:
        plot_tucker_summary(tucker_results, labels, categories)

    # Compare content-only vs position-conditioned for the most interesting head
    print("\n--- Content-only vs position-conditioned comparison ---")
    for prof in profiles:
        norms = prof["norms"]
        peak_d = norms.index(max(norms)) + 1
        head = prof["head"]
        layer = int(head[1])
        h = int(head[3])

        with torch.no_grad():
            T0, _, _, _, _ = build_content_only_tensor(model, layer, h)
            T_peak, _, _, _, _ = build_circuit_tensor_at_distance(model, layer, h, peak_d)

        # How different are they?
        diff_norm = (T_peak - T0).norm().item()
        cos_sim = (T_peak.flatten() @ T0.flatten()).item() / (T_peak.norm() * T0.norm() + 1e-8).item()
        print(f"  {head}: peak d={peak_d}, ||T_peak - T_content||={diff_norm:.3f}, "
              f"cos_sim={cos_sim:.3f}")

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

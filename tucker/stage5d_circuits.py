"""Stage 5d: Circuit-level view of BilinearGPT via Tucker decomposition.

The Tucker decomposition of T_d[q,k,o] gives us CIRCUITS, not just heads.
Each circuit is defined by:
  - G[i,i,i]: strength (how important this circuit is)
  - A[:,i]: query factor (which tokens at the current position activate this circuit)
  - B[:,i]: key factor (which tokens at the attended position activate this circuit)
  - C[:,i]: output factor (what output logits this circuit produces)

Reading a circuit: "When a token matching the query factor appears, and it
attends to a token matching the key factor (at distance d), this circuit
adds the output factor (scaled by G[i,i,i]) to the logits."

This script:
1. Extracts all Tucker circuits from all heads at key distances
2. Organizes them by circuit (not by head)
3. Labels each with its likely rule
4. Shows a circuit catalog figure

Run: python tucker/stage5d_circuits.py
"""

# %% Imports

import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "attn_circuits"))

from model import BilinearGPT, Config
from tucker_pipeline import tucker_pipeline
from stage5b_position_conditioned import (
    load_model, get_token_labels, extract_head_weights,
    build_circuit_tensor_at_distance,
)

FIGURES_DIR = Path(__file__).parent / "figures" / "stage5d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# %% Data structures

@dataclass
class Circuit:
    head: str           # e.g. "L0H0"
    layer: int
    head_idx: int
    distance: int       # relative distance d
    component: int      # Tucker component index
    strength: float     # G[i,i,i]
    query_factor: torch.Tensor   # (vocab,) — query direction
    key_factor: torch.Tensor     # (vocab,) — key direction
    output_factor: torch.Tensor  # (vocab,) — output direction
    recon_error: float
    rank: int

    def top_tokens(self, factor, labels, k=5):
        """Top-k tokens by absolute value for a factor."""
        topk = factor.abs().topk(k)
        return [(labels[i], factor[i].item()) for i in topk.indices]

    def summary(self, labels, categories):
        """One-line summary of this circuit."""
        q_top = labels[self.query_factor.abs().argmax()]
        k_top = labels[self.key_factor.abs().argmax()]
        o_top = labels[self.output_factor.abs().argmax()]
        q_cat = categories.get(self.query_factor.abs().argmax().item(), "?")
        k_cat = categories.get(self.key_factor.abs().argmax().item(), "?")
        o_cat = categories.get(self.output_factor.abs().argmax().item(), "?")
        return (f"{self.head} d={self.distance} c{self.component}: "
                f"{q_top}({q_cat}) × {k_top}({k_cat}) → {o_top}({o_cat}) "
                f"[G={self.strength:.1f}]")


# %% Extract all circuits

def extract_circuits(model, labels, categories, distances=[1, 2, 3, 5, 8]):
    """Extract Tucker circuits from all heads at specified distances."""
    circuits = []
    n_layers = model.config.n_layer
    n_heads = model.config.n_head

    for layer in range(n_layers):
        for head in range(n_heads):
            head_label = f"L{layer}H{head}"
            for d in distances:
                with torch.no_grad():
                    T_d, _, _, _, _ = build_circuit_tensor_at_distance(
                        model, layer, head, d
                    )

                if T_d.norm() < 1e-6:
                    continue

                # Auto rank: try 1-5, pick smallest with recon < 0.15
                best_rank = None
                for r in range(1, 6):
                    try:
                        _, _, _, _, m = tucker_pipeline(T_d, r, fast=True)
                        if m["recon_error"] < 0.15:
                            best_rank = r
                            break
                    except Exception:
                        continue
                if best_rank is None:
                    best_rank = 3  # fallback

                G, A, B, C, metrics = tucker_pipeline(T_d, best_rank, fast=True)
                r = min(best_rank, *G.shape)

                for c in range(r):
                    g_val = G[c, c, c].item()
                    # Skip negligible circuits
                    diag_max = max(abs(G[i, i, i].item()) for i in range(r))
                    if abs(g_val) < 0.05 * diag_max:
                        continue

                    circuits.append(Circuit(
                        head=head_label, layer=layer, head_idx=head,
                        distance=d, component=c, strength=g_val,
                        query_factor=A[:, c].clone(),
                        key_factor=B[:, c].clone(),
                        output_factor=C[:, c].clone(),
                        recon_error=metrics["recon_error"],
                        rank=best_rank,
                    ))

    return circuits


# %% Classify circuits by rule

def classify_circuit(circ, labels, categories):
    """Guess which rule a circuit implements based on its factors."""
    q_top5 = circ.query_factor.abs().topk(5).indices.tolist()
    k_top5 = circ.key_factor.abs().topk(5).indices.tolist()
    o_top5 = circ.output_factor.abs().topk(5).indices.tolist()

    q_cats = [categories.get(i, "?") for i in q_top5]
    k_cats = [categories.get(i, "?") for i in k_top5]
    o_cats = [categories.get(i, "?") for i in o_top5]

    q_top1 = labels[q_top5[0]]
    k_top1 = labels[k_top5[0]]
    o_top1 = labels[o_top5[0]]

    # Check for bracket circuits
    if k_top1 == "(" and o_cats[0] == "ADJ":
        return "paren_content"
    if k_top1 == "[" and o_cats[0] in ("VERB_I", "FUNC"):
        return "quote_content"
    if o_top1 == "]" and k_top1 == "[":
        return "bracket_close_quote"
    if o_top1 == ")" and k_top1 == "(":
        return "bracket_close_paren"
    if o_top1 == "]":
        return "bracket_close_quote"
    if o_top1 == ")":
        return "bracket_close_paren"

    # Category-level patterns
    q_dom = max(set(q_cats[:3]), key=q_cats[:3].count)
    k_dom = max(set(k_cats[:3]), key=k_cats[:3].count)
    o_dom = max(set(o_cats[:3]), key=o_cats[:3].count)

    # Trigram-like: specific tokens
    if q_dom == "NOUN" and k_dom == "NOUN" and o_dom == "VERB_T":
        return "cat_trigram" if circ.distance <= 3 else "skip_bigram"
    if q_dom == "ADJ" and k_dom == "PLACE" and o_dom == "FUNC":
        return "place_bigram_circuit"
    if q_dom == "NOUN" and o_dom == "VERB_T":
        return "noun_bigram_circuit"
    if q_dom == "ADJ" and o_dom in ("VERB_T", "NOUN"):
        return "cat_bigram_circuit"

    # Fallback: describe by categories
    return f"{q_dom}×{k_dom}→{o_dom}"


# %% Plotting

CAT_COLORS = {
    "NOUN": "#1f77b4", "PLACE": "#ff7f0e", "VERB_T": "#2ca02c",
    "VERB_I": "#d62728", "ADJ": "#9467bd", "FUNC": "#8c564b",
    "PUNCT": "#7f7f7f", "OPEN": "#bcbd22", "CLOSE": "#17becf",
    "STRUCT": "#aaaaaa",
}


def plot_circuit_card(circ, labels, categories, rule_label, ax_q, ax_k, ax_o):
    """Plot one circuit as three bar charts (query, key, output factors).

    Only show tokens with |factor value| > 10% of the max for that factor.
    """
    for ax, factor, role in [
        (ax_q, circ.query_factor, "Query (current pos)"),
        (ax_k, circ.key_factor, "Key (attended pos)"),
        (ax_o, circ.output_factor, "Output (logit)"),
    ]:
        vals = factor.numpy()
        threshold = 0.1 * np.abs(vals).max()
        mask = np.abs(vals) > threshold
        indices = np.where(mask)[0]

        if len(indices) == 0:
            ax.set_visible(False)
            continue

        filtered_vals = vals[indices]
        filtered_labels = [labels[i] for i in indices]
        colors = [CAT_COLORS.get(categories.get(int(i), "STRUCT"), "#aaa") for i in indices]

        bars = ax.barh(range(len(indices)), filtered_vals, color=colors, height=0.7)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(filtered_labels, fontsize=7)
        ax.set_title(role, fontsize=7, pad=2)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.tick_params(axis='x', labelsize=6)
        ax.invert_yaxis()


def plot_circuit_catalog(circuits, labels, categories, max_per_page=8):
    """Plot a catalog of circuits, sorted by strength."""
    # Sort by absolute strength
    sorted_circs = sorted(circuits, key=lambda c: -abs(c.strength))

    # Deduplicate: same head + same top tokens at different distances → keep strongest
    seen = set()
    unique_circs = []
    for circ in sorted_circs:
        q_top = labels[circ.query_factor.abs().argmax()]
        k_top = labels[circ.key_factor.abs().argmax()]
        o_top = labels[circ.output_factor.abs().argmax()]
        key = (circ.head, q_top, k_top, o_top)
        if key not in seen:
            seen.add(key)
            unique_circs.append(circ)

    n_circuits = min(len(unique_circs), max_per_page * 3)  # max 3 pages
    pages = (n_circuits + max_per_page - 1) // max_per_page

    for page in range(pages):
        start = page * max_per_page
        end = min(start + max_per_page, n_circuits)
        page_circs = unique_circs[start:end]
        n = len(page_circs)

        fig = plt.figure(figsize=(16, 2.5 * n))
        outer_gs = gridspec.GridSpec(n, 1, hspace=0.6)

        for row, circ in enumerate(page_circs):
            rule = classify_circuit(circ, labels, categories)
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[row],
                                                        width_ratios=[1, 1, 1, 0.3],
                                                        wspace=0.4)
            ax_q = fig.add_subplot(inner_gs[0])
            ax_k = fig.add_subplot(inner_gs[1])
            ax_o = fig.add_subplot(inner_gs[2])
            ax_info = fig.add_subplot(inner_gs[3])

            plot_circuit_card(circ, labels, categories, rule, ax_q, ax_k, ax_o)

            # Info panel
            ax_info.axis("off")
            info_text = (
                f"{circ.head} d={circ.distance}\n"
                f"G = {circ.strength:.1f}\n"
                f"rank = {circ.rank}\n"
                f"recon = {circ.recon_error:.3f}\n"
                f"\nRule: {rule}"
            )
            ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes,
                         fontsize=8, verticalalignment="center", fontfamily="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        fig.suptitle(f"Circuit Catalog (page {page + 1}/{pages}, sorted by |G|)",
                     fontsize=13, fontweight="bold", y=1.0)
        plt.savefig(FIGURES_DIR / f"circuit_catalog_p{page + 1}.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"Saved: circuit_catalog_p{page + 1}.png")


# %% Summary table: circuits grouped by rule

def print_circuit_table(circuits, labels, categories):
    """Print circuits grouped by rule type."""
    # Classify all
    by_rule = defaultdict(list)
    for circ in circuits:
        rule = classify_circuit(circ, labels, categories)
        by_rule[rule].append(circ)

    print("\n" + "=" * 80)
    print("CIRCUIT CATALOG BY RULE")
    print("=" * 80)

    for rule in sorted(by_rule.keys()):
        circs = sorted(by_rule[rule], key=lambda c: -abs(c.strength))
        print(f"\n--- {rule} ({len(circs)} circuits) ---")
        # Show top 5
        for circ in circs[:5]:
            q_toks = circ.top_tokens(circ.query_factor, labels, k=3)
            k_toks = circ.top_tokens(circ.key_factor, labels, k=3)
            o_toks = circ.top_tokens(circ.output_factor, labels, k=3)
            q_str = ", ".join(f"{t}({v:+.2f})" for t, v in q_toks)
            k_str = ", ".join(f"{t}({v:+.2f})" for t, v in k_toks)
            o_str = ", ".join(f"{t}({v:+.2f})" for t, v in o_toks)
            print(f"  {circ.head} d={circ.distance}: G={circ.strength:>8.1f} | "
                  f"Q=[{q_str}] K=[{k_str}] → O=[{o_str}]")


# %% Rule coverage: which rules have Tucker circuits, which don't?

def rule_coverage(circuits, labels, categories):
    """Check which known rules have corresponding Tucker circuits."""
    by_rule = defaultdict(list)
    for circ in circuits:
        rule = classify_circuit(circ, labels, categories)
        by_rule[rule].append(circ)

    known_rules = [
        "paren_content", "quote_content", "bracket_close_quote", "bracket_close_paren",
        "cat_trigram", "noun_bigram_circuit", "place_bigram_circuit",
        "cat_bigram_circuit", "skip_bigram",
    ]

    print("\n" + "=" * 80)
    print("RULE COVERAGE")
    print("=" * 80)

    found = []
    missing = []
    for rule in known_rules:
        circs = by_rule.get(rule, [])
        if circs:
            best = max(circs, key=lambda c: abs(c.strength))
            found.append((rule, best))
            print(f"  FOUND  {rule:<30} best: {best.head} d={best.distance} G={best.strength:.1f}")
        else:
            missing.append(rule)
            print(f"  MISSING {rule:<30}")

    # Rules not in known list
    other = [r for r in by_rule if r not in known_rules]
    if other:
        print(f"\n  Other circuit types found:")
        for rule in sorted(other):
            n = len(by_rule[rule])
            best = max(by_rule[rule], key=lambda c: abs(c.strength))
            print(f"    {rule:<30} ({n} circuits, strongest G={best.strength:.1f})")

    print(f"\n  Rules with direct-path only (no attention circuit needed):")
    print(f"    cat_bigram, noun_bigram, place_bigram — handled by lm_head @ embed")
    print(f"    paren_open, quote_open — unpredictable (random bracket insertions)")

    return found, missing


# %% Plot: single head deep-dive using factors

def plot_head_circuits(circuits, head_label, labels, categories):
    """For one head, show all its circuits across distances."""
    head_circs = [c for c in circuits if c.head == head_label]
    if not head_circs:
        return

    # Group by distance
    by_dist = defaultdict(list)
    for c in head_circs:
        by_dist[c.distance].append(c)

    distances = sorted(by_dist.keys())
    max_comp = max(len(v) for v in by_dist.values())

    fig = plt.figure(figsize=(5 * len(distances), 3 * max_comp))
    outer_gs = gridspec.GridSpec(max_comp, len(distances), hspace=0.5, wspace=0.4)

    for col, d in enumerate(distances):
        circs = sorted(by_dist[d], key=lambda c: -abs(c.strength))
        for row, circ in enumerate(circs):
            rule = classify_circuit(circ, labels, categories)
            inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[row, col],
                                                     wspace=0.3)
            ax_q = fig.add_subplot(inner[0])
            ax_k = fig.add_subplot(inner[1])
            ax_o = fig.add_subplot(inner[2])
            plot_circuit_card(circ, labels, categories, rule, ax_q, ax_k, ax_o)

            # Label
            ax_q.set_ylabel(f"d={d} G={circ.strength:.0f}\n{rule}", fontsize=7)

    fig.suptitle(f"{head_label}: Tucker circuits by distance", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.savefig(FIGURES_DIR / f"head_{head_label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: head_{head_label}.png")


# %% Main

def main():
    print("Loading model...")
    model, config = load_model()
    labels, categories = get_token_labels()
    print(f"Model: {config.n_layer}L {config.n_head}H d={config.n_embd}, vocab={config.vocab_size}")

    # Extract all circuits
    print("\nExtracting Tucker circuits from all heads at d=1,2,3,5,8...")
    circuits = extract_circuits(model, labels, categories, distances=[1, 2, 3, 5, 8])
    print(f"Found {len(circuits)} circuits total")

    # Print organized table
    print_circuit_table(circuits, labels, categories)

    # Coverage analysis
    rule_coverage(circuits, labels, categories)

    # Circuit catalog (the main figure)
    print("\nPlotting circuit catalog...")
    plot_circuit_catalog(circuits, labels, categories, max_per_page=10)

    # Per-head deep dives
    print("\nPlotting per-head circuit views...")
    for layer in range(config.n_layer):
        for head in range(config.n_head):
            plot_head_circuits(circuits, f"L{layer}H{head}", labels, categories)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

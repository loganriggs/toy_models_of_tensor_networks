# %%
"""
Automated circuit analysis for v3 BilinearGPT (2L 4H d=64).

Connects model weights to specific learned behaviors using the exact
decomposition: logits = lm_head(emb) + sum(lm_head(L0_head_h)) + sum(lm_head(L1_head_h)).
This is exact because there are no norms, no MLP, and lm_head has no bias.

Phases:
  1. Head ablation — which heads matter for which rules
  2. Circuit extraction & category-grouped visualization
  3. Automated circuit-rule tests
  4. Logit attribution decomposition (exact per-head contributions)
  5. Attention pattern analysis at rule-fire positions
"""
import sys
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import einsum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BilinearGPT, Config, apply_rotary_emb
from generator_v3 import (
    LanguageV3, make_batch, per_rule_loss, true_entropies,
    CAT_ORDER, TRANSITION_MATRIX, BRACKET_TYPES, NOUN_BIGRAM_PROBS,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# === CONFIG ===
MODEL_TAG = 'v3_all_2L_4H_d64_5000steps'
MODEL_PATH = os.path.join(ROOT_DIR, 'models', f'bilinear_gpt_{MODEL_TAG}.pt')
CONFIG_PATH = os.path.join(ROOT_DIR, 'configs', f'bilinear_gpt_{MODEL_TAG}.json')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', MODEL_TAG)
os.makedirs(FIG_DIR, exist_ok=True)

# %%
# === Load model and generator ===
import json
with open(CONFIG_PATH, 'r') as f:
    config_dict = json.load(f)

config = Config(
    vocab_size=config_dict['vocab_size'],
    n_layer=config_dict['n_layer'],
    n_head=config_dict['n_head'],
    n_embd=config_dict['n_embd'],
    seq_len=config_dict['seq_len'],
    block_has_mlp=tuple(config_dict.get('block_has_mlp', [False] * config_dict['n_layer'])),
    use_rmsnorm=config_dict.get('use_rmsnorm', False),
    use_qk_norm=config_dict.get('use_qk_norm', False),
    use_final_norm=config_dict.get('use_final_norm', False),
)

model = BilinearGPT(config).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

gen = LanguageV3(seed=42)
print(f"Loaded {MODEL_TAG}: {model.count_params():,} params, {config.n_layer}L {config.n_head}H d={config.n_embd}")
print(f"Vocab: {config.vocab_size} tokens, device={device}")

# Vocab info for category-grouped plots
N_TOK = gen.tokens_per_category  # 8
CAT_NAMES = ['NOUN', 'PLACE', 'VERB_T', 'VERB_I', 'ADJ', 'FUNC', 'STRUCT']
CAT_BOUNDS = [0, N_TOK, 2*N_TOK, 3*N_TOK, 4*N_TOK, 5*N_TOK, 6*N_TOK, config.vocab_size]
TOKEN_LABELS = [gen.id2token[i] for i in range(config.vocab_size)]
N_HEAD = config.n_head
N_LAYER = config.n_layer
D_HEAD = config.n_embd // N_HEAD
HEAD_LABELS = [f'L{l}H{h}' for l in range(N_LAYER) for h in range(N_HEAD)]

# Eval batches (reused across phases)
EVAL_BATCHES = 30
EVAL_BS = 128
EVAL_SEQ = config.seq_len + 1

# %%
# ========================================================================
# PHASE 1: HEAD ABLATION
# ========================================================================
print("\n" + "=" * 70)
print("PHASE 1: HEAD ABLATION")
print("=" * 70)


@contextmanager
def ablate_head(model, layer, head):
    """Zero out a single head's output projection. Saves/restores weights."""
    attn = model.blocks[layer].attn
    sl = slice(head * D_HEAD, (head + 1) * D_HEAD)
    saved = attn.out.weight.data[:, sl].clone()
    attn.out.weight.data[:, sl] = 0
    try:
        yield
    finally:
        attn.out.weight.data[:, sl] = saved


def compute_eval_losses(model, gen, n_batches=EVAL_BATCHES, batch_size=EVAL_BS, seq_len=EVAL_SEQ):
    """Compute per-rule CE losses over eval batches."""
    eval_gen = LanguageV3(seed=999)
    all_rule_losses = defaultdict(list)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, labels, mask = make_batch(eval_gen, batch_size, seq_len, device=device)
            logits, _ = model(x, targets=y)
            rl = per_rule_loss(logits, y, labels, mask)
            for k, v in rl.items():
                all_rule_losses[k].append(v)
    return {k: np.mean(v) for k, v in all_rule_losses.items()}


# Baseline losses
baseline_losses = compute_eval_losses(model, gen)
print("\nBaseline per-rule CE losses:")
h_true = true_entropies(gen)
for rule in sorted(baseline_losses, key=baseline_losses.get):
    ce = baseline_losses[rule]
    kl = ce - h_true.get(rule, 0)
    print(f"  {rule:<20s}  CE={ce:.4f}  KL={kl:.4f}")

# No-attention baseline (direct path only — ablate ALL heads)
print("\nComputing no-attention baseline (all heads ablated)...")
from contextlib import ExitStack
with ExitStack() as stack:
    for layer in range(N_LAYER):
        for head in range(N_HEAD):
            stack.enter_context(ablate_head(model, layer, head))
    no_attn_losses = compute_eval_losses(model, gen)

print("\nAttention benefit per rule (no_attn_CE - full_CE):")
attn_benefit = {}
for rule in sorted(baseline_losses, key=baseline_losses.get):
    benefit = no_attn_losses.get(rule, 0) - baseline_losses.get(rule, 0)
    attn_benefit[rule] = benefit
    print(f"  {rule:<20s}  no_attn={no_attn_losses.get(rule,0):.3f}  full={baseline_losses[rule]:.3f}  benefit={benefit:+.3f}")

# Ablation sweep
delta_losses = {}  # head_label -> {rule: delta_ce}
for layer in range(N_LAYER):
    for head in range(N_HEAD):
        label = f'L{layer}H{head}'
        print(f"  Ablating {label}...", end="", flush=True)
        with ablate_head(model, layer, head):
            ablated = compute_eval_losses(model, gen)
        delta = {r: ablated.get(r, 0) - baseline_losses.get(r, 0) for r in baseline_losses}
        delta_losses[label] = delta
        top = max(delta, key=delta.get)
        print(f" biggest impact: {top} (+{delta[top]:.4f})")

# %%
# --- Compute percentage-normalized ablation ---
# For each rule: % of total positive ablation impact from each head
# This shows the distribution of responsibility across heads

rule_names = sorted(baseline_losses.keys())
head_names = HEAD_LABELS

delta_matrix = np.array([[delta_losses[h].get(r, 0) for r in rule_names] for h in head_names])

# Percentage: each head's share of total positive ΔCE for that rule
# (how much of the "attention helps this rule" story is this head?)
pos_delta = np.maximum(delta_matrix, 0)
col_sums = pos_delta.sum(axis=0, keepdims=True)
col_sums = np.where(col_sums > 0, col_sums, 1)  # avoid div by zero
pct_matrix = pos_delta / col_sums * 100

# Also compute: % of total attention benefit (can exceed 100% due to interactions)
pct_of_benefit = np.zeros_like(delta_matrix)
for j, rule in enumerate(rule_names):
    benefit = attn_benefit.get(rule, 0)
    if benefit > 0.01:  # only normalize when attention actually helps
        pct_of_benefit[:, j] = delta_matrix[:, j] / benefit * 100

# --- Plot: percentage of attention benefit per head ---
fig, ax = plt.subplots(figsize=(max(10, len(rule_names) * 0.8), 5))
im = ax.imshow(pct_matrix, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(rule_names)))
ax.set_xticklabels(rule_names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(head_names)))
ax.set_yticklabels(head_names, fontsize=10)
ax.set_xlabel('Rule')
ax.set_ylabel('Head')
ax.set_title(f'Head Ablation: % of total positive impact per rule — {MODEL_TAG}')
fig.colorbar(im, ax=ax, shrink=0.8, label='% of rule impact')

for i in range(len(head_names)):
    for j in range(len(rule_names)):
        val = pct_matrix[i, j]
        if val > 3:
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=7,
                    color='white' if val > 50 else 'black')

fig.tight_layout()
path = os.path.join(FIG_DIR, 'ablation_heatmap_pct.png')
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"\nSaved: {path}")

# Also save the raw ΔCE heatmap for reference
fig, ax = plt.subplots(figsize=(max(10, len(rule_names) * 0.8), 5))
vmax = np.abs(delta_matrix).max()
im = ax.imshow(delta_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(rule_names)))
ax.set_xticklabels(rule_names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(head_names)))
ax.set_yticklabels(head_names, fontsize=10)
ax.set_xlabel('Rule')
ax.set_ylabel('Head')
ax.set_title(f'Head Ablation: raw ΔCE when head zeroed — {MODEL_TAG}')
fig.colorbar(im, ax=ax, shrink=0.8, label='ΔCE (nats)')
for i in range(len(head_names)):
    for j in range(len(rule_names)):
        val = delta_matrix[i, j]
        if abs(val) > 0.05:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(val) > vmax * 0.6 else 'black')
fig.tight_layout()
path = os.path.join(FIG_DIR, 'ablation_heatmap_raw.png')
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# --- Rule specialization table ---
# For each rule: show distribution of responsibility + concentration metric
print("\n" + "-" * 80)
print("ABLATION: responsibility distribution per rule")
print("-" * 80)
print(f"{'Rule':<20s} {'Top head':>8s} {'Top%':>5s} {'Top2%':>6s} {'Conc':>5s}  Distribution (heads with >5%)")
print("-" * 80)
for j, rule in enumerate(rule_names):
    pcts = pct_matrix[:, j]
    sorted_idx = np.argsort(pcts)[::-1]
    top1_pct = pcts[sorted_idx[0]]
    top2_pct = pcts[sorted_idx[0]] + pcts[sorted_idx[1]]
    # HHI concentration: 10000 = monopoly (one head), ~1250 = uniform across 8
    hhi = sum(p**2 for p in pcts) / 100  # normalized so max=100
    top_head = head_names[sorted_idx[0]]

    dist_parts = []
    for idx in sorted_idx:
        if pcts[idx] > 5:
            dist_parts.append(f"{head_names[idx]}:{pcts[idx]:.0f}%")

    specialization = "FOCUSED" if top1_pct > 60 else ("SHARED" if top1_pct < 35 else "MIXED")
    print(f"{rule:<20s} {top_head:>8s} {top1_pct:>4.0f}% {top2_pct:>5.0f}% {hhi:>5.0f}  {' '.join(dist_parts)}  [{specialization}]")


# %%
# ========================================================================
# PHASE 2: CIRCUIT EXTRACTION & VISUALIZATION
# ========================================================================
print("\n" + "=" * 70)
print("PHASE 2: CIRCUIT EXTRACTION & VISUALIZATION")
print("=" * 70)


def extract_circuits(model, layer=0):
    """Extract per-head QK, OV circuits and direct path (content-only, no RoPE)."""
    attn = model.blocks[layer].attn
    E = model.embed.weight.detach().float()
    U = model.lm_head.weight.detach().float()

    qk_circuits, ov_circuits = [], []
    for h in range(N_HEAD):
        sl = slice(h * D_HEAD, (h + 1) * D_HEAD)

        W_Q1 = attn.q1.weight[sl].detach().float()
        W_K1 = attn.k1.weight[sl].detach().float()
        W_Q2 = attn.q2.weight[sl].detach().float()
        W_K2 = attn.k2.weight[sl].detach().float()

        QK1 = E @ W_Q1.T @ W_K1 @ E.T
        QK2 = E @ W_Q2.T @ W_K2 @ E.T
        QK_combined = (QK1 * QK2) / (D_HEAD ** 2)
        qk_circuits.append(QK_combined)

        W_V = attn.v.weight[sl].detach().float()
        W_O = attn.out.weight[:, sl].detach().float()
        OV = E @ W_V.T @ W_O.T @ U.T
        ov_circuits.append(OV)

    direct = E @ U.T
    return qk_circuits, ov_circuits, direct


# Extract for both layers
circuits = {}
for layer in range(N_LAYER):
    qk, ov, direct = extract_circuits(model, layer)
    circuits[f'L{layer}_qk'] = qk
    circuits[f'L{layer}_ov'] = ov
circuits['direct'] = direct
print(f"Extracted circuits for {N_LAYER} layers × {N_HEAD} heads")


def plot_category_grouped(mat, title, save_name, cmap='RdBu_r'):
    """Plot a V×V matrix with category boundary lines."""
    fig, ax = plt.subplots(figsize=(8, 7))
    data = mat.cpu().numpy() if torch.is_tensor(mat) else mat
    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')

    # Category boundaries
    for b in CAT_BOUNDS[1:-1]:
        ax.axhline(b - 0.5, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(b - 0.5, color='black', linewidth=0.5, alpha=0.5)

    # Category labels at midpoints
    mids = [(CAT_BOUNDS[i] + CAT_BOUNDS[i+1]) / 2 for i in range(len(CAT_NAMES))]
    ax.set_xticks(mids)
    ax.set_xticklabels(CAT_NAMES, fontsize=8)
    ax.set_yticks(mids)
    ax.set_yticklabels(CAT_NAMES, fontsize=8)
    ax.set_xlabel('Key / Source')
    ax.set_ylabel('Query / Dest')
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compute_category_circuit(mat):
    """Average a V×V matrix within category blocks → (n_cats × n_cats)."""
    data = mat.cpu().numpy() if torch.is_tensor(mat) else mat
    n_cats = len(CAT_NAMES)
    result = np.zeros((n_cats, n_cats))
    for i in range(n_cats):
        for j in range(n_cats):
            block = data[CAT_BOUNDS[i]:CAT_BOUNDS[i+1], CAT_BOUNDS[j]:CAT_BOUNDS[j+1]]
            result[i, j] = block.mean()
    return result


# Direct path
p = plot_category_grouped(circuits['direct'], f'Direct path: E @ U^T — {MODEL_TAG}', 'circuit_direct.png')
print(f"Saved: {p}")

# Per-head QK and OV for each layer
for layer in range(N_LAYER):
    # QK
    fig, axes = plt.subplots(1, N_HEAD, figsize=(4 * N_HEAD, 4))
    for h in range(N_HEAD):
        data = circuits[f'L{layer}_qk'][h].cpu().numpy()
        vmax = max(abs(data.min()), abs(data.max()))
        im = axes[h].imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        for b in CAT_BOUNDS[1:-1]:
            axes[h].axhline(b - 0.5, color='black', linewidth=0.3, alpha=0.3)
            axes[h].axvline(b - 0.5, color='black', linewidth=0.3, alpha=0.3)
        axes[h].set_title(f'L{layer}H{h} QK', fontsize=9)
        fig.colorbar(im, ax=axes[h], shrink=0.6)
    fig.suptitle(f'Bilinear QK Circuits (content-only) — Layer {layer}', fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, f'circuit_qk_L{layer}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # OV
    fig, axes = plt.subplots(1, N_HEAD, figsize=(4 * N_HEAD, 4))
    for h in range(N_HEAD):
        data = circuits[f'L{layer}_ov'][h].cpu().numpy()
        vmax = max(abs(data.min()), abs(data.max()))
        im = axes[h].imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        for b in CAT_BOUNDS[1:-1]:
            axes[h].axhline(b - 0.5, color='black', linewidth=0.3, alpha=0.3)
            axes[h].axvline(b - 0.5, color='black', linewidth=0.3, alpha=0.3)
        axes[h].set_title(f'L{layer}H{h} OV', fontsize=9)
        fig.colorbar(im, ax=axes[h], shrink=0.6)
    fig.suptitle(f'OV Circuits — Layer {layer}', fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, f'circuit_ov_L{layer}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

# Category-reduced circuits
print("\nCategory-reduced direct path:")
cat_direct = compute_category_circuit(circuits['direct'])
header = "          " + " ".join(f"{c:>7s}" for c in CAT_NAMES)
print(header)
for i, cat in enumerate(CAT_NAMES):
    row = " ".join(f"{cat_direct[i, j]:>7.3f}" for j in range(len(CAT_NAMES)))
    print(f"  {cat:8s} {row}")


# %%
# ========================================================================
# PHASE 2b: ZOOMED CIRCUIT ANALYSIS PER RULE
# ========================================================================
# For each rule type, extract the specific token submatrices from QK/OV
# and show what the responsible head actually encodes.
print("\n" + "=" * 70)
print("PHASE 2b: ZOOMED CIRCUIT ANALYSIS PER RULE")
print("=" * 70)


def plot_submatrix(mat, row_tokens, col_tokens, title, save_name, annotate=True):
    """Plot a submatrix extracted from a full V×V circuit matrix."""
    if torch.is_tensor(mat):
        mat = mat.cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(4, len(col_tokens) * 0.55), max(3, len(row_tokens) * 0.45)))
    vmax = max(abs(mat.min()), abs(mat.max()), 0.01)
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(col_tokens)))
    ax.set_xticklabels(col_tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_tokens)))
    ax.set_yticklabels(row_tokens, fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    if annotate and len(row_tokens) <= 16 and len(col_tokens) <= 16:
        for i in range(len(row_tokens)):
            for j in range(len(col_tokens)):
                val = mat[i, j]
                if abs(val) > vmax * 0.15:
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6,
                            color='white' if abs(val) > vmax * 0.5 else 'black')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def extract_submat(full_mat, row_ids, col_ids):
    """Extract a submatrix from a V×V matrix given row and column token IDs."""
    if torch.is_tensor(full_mat):
        full_mat = full_mat.cpu().numpy()
    return full_mat[np.ix_(row_ids, col_ids)]


# Helper: token IDs for each category
cat_ids = {}
for cat_name, bound_start, bound_end in zip(CAT_NAMES, CAT_BOUNDS[:-1], CAT_BOUNDS[1:]):
    cat_ids[cat_name] = list(range(bound_start, bound_end))

nouns = gen.categories['NOUN']
noun_ids = [gen.token2id[t] for t in nouns]
places = gen.categories['PLACE']
place_ids = [gen.token2id[t] for t in places]
verb_ts = gen.categories['VERB_T']
verb_t_ids = [gen.token2id[t] for t in verb_ts]
verb_is = gen.categories['VERB_I']
verb_i_ids = [gen.token2id[t] for t in verb_is]
funcs = gen.categories['FUNC']
func_ids = [gen.token2id[t] for t in funcs]
adjs = gen.categories['ADJ']
adj_ids = [gen.token2id[t] for t in adjs]


def plot_all_heads_grid(circuit_type, row_ids, col_ids, row_labels, col_labels,
                        suptitle, save_name, annotate_thresh=0.15):
    """Plot a 2×4 grid (layers × heads) of zoomed submatrices.

    Args:
        circuit_type: 'qk' or 'ov'
        row_ids, col_ids: token IDs for rows/cols of the submatrix
        row_labels, col_labels: token names
        suptitle: figure title
        save_name: filename
    """
    fig, axes = plt.subplots(N_LAYER, N_HEAD,
        figsize=(3.5 * N_HEAD + 1.2, 3 * N_LAYER + 0.5),
        squeeze=False, constrained_layout=True)

    # Use shared colorscale across all panels
    all_vals = []
    for layer in range(N_LAYER):
        for head in range(N_HEAD):
            mat = circuits[f'L{layer}_{circuit_type}'][head]
            sub = extract_submat(mat, row_ids, col_ids)
            all_vals.append(sub)
    global_vmax = max(max(abs(s.min()), abs(s.max())) for s in all_vals)
    global_vmax = max(global_vmax, 0.01)

    for layer in range(N_LAYER):
        for head in range(N_HEAD):
            ax = axes[layer][head]
            sub = all_vals[layer * N_HEAD + head]
            im = ax.imshow(sub, cmap='RdBu_r', vmin=-global_vmax, vmax=global_vmax, aspect='auto')
            ax.set_title(f'L{layer}H{head}', fontsize=9)

            if head == 0:
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=7)
            else:
                ax.set_yticks([])

            if layer == N_LAYER - 1:
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
            else:
                ax.set_xticks([])

            # Annotate strong values
            if len(row_labels) <= 16 and len(col_labels) <= 16:
                for i in range(len(row_labels)):
                    for j in range(len(col_labels)):
                        val = sub[i, j]
                        if abs(val) > global_vmax * annotate_thresh:
                            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=5,
                                    color='white' if abs(val) > global_vmax * 0.5 else 'black')

    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label='Value')
    fig.suptitle(suptitle, fontsize=11)
    path = os.path.join(FIG_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_full_circuit_grid(query_ids, key_ids, output_ids,
                           query_labels, key_labels, output_labels,
                           suptitle, save_name):
    """Plot QK×OV full circuit for all heads in a 2×4 grid."""
    fig, axes = plt.subplots(N_LAYER, N_HEAD,
        figsize=(3.5 * N_HEAD + 1.2, 3 * N_LAYER + 0.5),
        squeeze=False, constrained_layout=True)

    all_full = []
    for layer in range(N_LAYER):
        for head in range(N_HEAD):
            qk = extract_submat(circuits[f'L{layer}_qk'][head], query_ids, key_ids)
            ov = extract_submat(circuits[f'L{layer}_ov'][head], key_ids, output_ids)
            full = qk @ ov
            all_full.append(full)
    global_vmax = max(max(abs(f.min()), abs(f.max())) for f in all_full)
    global_vmax = max(global_vmax, 0.01)

    for layer in range(N_LAYER):
        for head in range(N_HEAD):
            ax = axes[layer][head]
            full = all_full[layer * N_HEAD + head]
            im = ax.imshow(full, cmap='RdBu_r', vmin=-global_vmax, vmax=global_vmax, aspect='auto')
            ax.set_title(f'L{layer}H{head}', fontsize=9)

            if head == 0:
                ax.set_yticks(range(len(query_labels)))
                ax.set_yticklabels(query_labels, fontsize=7)
            else:
                ax.set_yticks([])

            if layer == N_LAYER - 1:
                ax.set_xticks(range(len(output_labels)))
                ax.set_xticklabels(output_labels, rotation=45, ha='right', fontsize=7)
            else:
                ax.set_xticks([])

            if len(query_labels) <= 16 and len(output_labels) <= 16:
                for i in range(len(query_labels)):
                    for j in range(len(output_labels)):
                        val = full[i, j]
                        if abs(val) > global_vmax * 0.15:
                            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=5,
                                    color='white' if abs(val) > global_vmax * 0.5 else 'black')

    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label='QK×OV')
    fig.suptitle(suptitle, fontsize=11)
    path = os.path.join(FIG_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --- 1. Direct path (no heads — just embed×unembed) ---
print("\n--- Direct path: place_bigram (PLACE → FUNC) ---")
sub = extract_submat(circuits['direct'], place_ids, func_ids)
p = plot_submatrix(sub, places, funcs,
    'Direct path: PLACE→FUNC (place bigrams)', 'zoom_direct_place_func.png')
print(f"Saved: {p}")
for place_tok, func_tok in gen.place_bigrams.items():
    pid, fid = gen.token2id[place_tok], gen.token2id[func_tok]
    val = circuits['direct'].cpu().numpy()[pid, fid]
    row = circuits['direct'].cpu().numpy()[pid]
    rank = (row >= val).sum()
    print(f"  {place_tok}→{func_tok}: direct={val:.3f} (rank {rank}/{config.vocab_size})")

print("\n--- Direct path: noun_bigram (NOUN → VERB_T) ---")
sub = extract_submat(circuits['direct'], noun_ids, verb_t_ids)
p = plot_submatrix(sub, nouns, verb_ts,
    'Direct path: NOUN→VERB_T (noun bigrams)', 'zoom_direct_noun_verbt.png')
print(f"Saved: {p}")
for noun_tok, dist in gen.noun_bigram_dists.items():
    nid = gen.token2id[noun_tok]
    row = circuits['direct'].cpu().numpy()[nid]
    verb_vals = [(v, row[gen.token2id[v]], p) for v, p in dist]
    verb_str = ", ".join(f"{v}={val:.2f}({p:.0%})" for v, val, p in verb_vals)
    print(f"  {noun_tok}: {verb_str}")


# --- 2. tok_trigram: all heads QK on NOUN×FUNC, OV on FUNC×VERB_I ---
# Rules like (the, alice→runs): QK[alice,the] should be high, OV[the,runs] should be high
print("\n--- tok_trigram: all heads QK (NOUN→FUNC) ---")
p = plot_all_heads_grid('qk', noun_ids, func_ids, nouns, funcs,
    f'QK circuits zoomed: NOUN queries × FUNC keys (tok_trigram) — {MODEL_TAG}',
    'zoom_allheads_qk_toktrigram.png')
print(f"Saved: {p}")

print("--- tok_trigram: all heads OV (FUNC→VERB_I) ---")
p = plot_all_heads_grid('ov', func_ids, verb_i_ids, funcs, verb_is,
    f'OV circuits zoomed: FUNC source → VERB_I output (tok_trigram) — {MODEL_TAG}',
    'zoom_allheads_ov_toktrigram.png')
print(f"Saved: {p}")

print("--- tok_trigram: all heads full circuit QK×OV (NOUN × FUNC → VERB_I) ---")
p = plot_full_circuit_grid(noun_ids, func_ids, verb_i_ids,
    nouns, funcs, verb_is,
    f'Full circuit QK×OV: NOUN × FUNC → VERB_I (tok_trigram) — {MODEL_TAG}',
    'zoom_allheads_full_toktrigram.png')
print(f"Saved: {p}")

# Print tok_trigram rule verification
for (prev2, prev), output in gen.tok_trigrams.items():
    if prev2 in funcs and prev in nouns and output in verb_is:
        qi = nouns.index(prev)
        oi = verb_is.index(output)
        # Check L0H0 (the known responsible head)
        qk_val = circuits['L0_qk'][0].cpu().numpy()[gen.token2id[prev], gen.token2id[prev2]]
        ov_val = circuits['L0_ov'][0].cpu().numpy()[gen.token2id[prev2], gen.token2id[output]]
        print(f"  {prev2},{prev}→{output}: L0H0 QK={qk_val:.2f}, OV={ov_val:.2f}")


# --- 3. skip_bigram: all heads QK on ADJ×PLACE, OV on PLACE×FUNC ---
# Rules like (beach...big→at): QK[big,beach] high, OV[beach,at] high
print("\n--- skip_bigram: all heads QK (ADJ→PLACE) ---")
p = plot_all_heads_grid('qk', adj_ids, place_ids, adjs, places,
    f'QK circuits zoomed: ADJ queries × PLACE keys (skip_bigram) — {MODEL_TAG}',
    'zoom_allheads_qk_skipbigram.png')
print(f"Saved: {p}")

print("--- skip_bigram: all heads OV (PLACE→FUNC) ---")
p = plot_all_heads_grid('ov', place_ids, func_ids, places, funcs,
    f'OV circuits zoomed: PLACE source → FUNC output (skip_bigram) — {MODEL_TAG}',
    'zoom_allheads_ov_skipbigram.png')
print(f"Saved: {p}")

print("--- skip_bigram: all heads full circuit QK×OV (ADJ × PLACE → FUNC) ---")
p = plot_full_circuit_grid(adj_ids, place_ids, func_ids,
    adjs, places, funcs,
    f'Full circuit QK×OV: ADJ × PLACE → FUNC (skip_bigram) — {MODEL_TAG}',
    'zoom_allheads_full_skipbigram.png')
print(f"Saved: {p}")

for (anchor, trigger), output in gen.skip_bigrams.items():
    qk_val = circuits['L1_qk'][1].cpu().numpy()[gen.token2id[trigger], gen.token2id[anchor]]
    ov_val = circuits['L1_ov'][1].cpu().numpy()[gen.token2id[anchor], gen.token2id[output]]
    print(f"  {anchor}...{trigger}→{output}: L1H1 QK={qk_val:.2f}, OV={ov_val:.2f}")


# --- 4. place_bigram: all heads QK and OV on PLACE×FUNC ---
# Direct path handles the mapping, but L1 heads amplify
print("\n--- place_bigram: all heads QK (PLACE×FUNC) ---")
p = plot_all_heads_grid('qk', place_ids, func_ids, places, funcs,
    f'QK circuits zoomed: PLACE × FUNC (place_bigram) — {MODEL_TAG}',
    'zoom_allheads_qk_placebigram.png')
print(f"Saved: {p}")

print("--- place_bigram: all heads OV (PLACE→FUNC) ---")
p = plot_all_heads_grid('ov', place_ids, func_ids, places, funcs,
    f'OV circuits zoomed: PLACE source → FUNC output (place_bigram) — {MODEL_TAG}',
    'zoom_allheads_ov_placebigram.png')
print(f"Saved: {p}")


# --- 5. paren_content: all heads on ADJ+NOUN tokens ---
adj_noun_ids = adj_ids + noun_ids
adj_noun_labels = adjs + nouns
print("\n--- paren_content: all heads QK (ADJ+NOUN × ADJ+NOUN) ---")
p = plot_all_heads_grid('qk', adj_noun_ids, adj_noun_ids, adj_noun_labels, adj_noun_labels,
    f'QK circuits zoomed: ADJ+NOUN (paren_content) — {MODEL_TAG}',
    'zoom_allheads_qk_paren.png')
print(f"Saved: {p}")

print("--- paren_content: all heads OV (ADJ+NOUN → ADJ+NOUN) ---")
p = plot_all_heads_grid('ov', adj_noun_ids, adj_noun_ids, adj_noun_labels, adj_noun_labels,
    f'OV circuits zoomed: ADJ+NOUN (paren_content) — {MODEL_TAG}',
    'zoom_allheads_ov_paren.png')
print(f"Saved: {p}")


# --- 6. quote_content: all heads on VERB_I+FUNC tokens ---
vi_func_ids = verb_i_ids + func_ids
vi_func_labels = verb_is + funcs
print("\n--- quote_content: all heads QK (VERB_I+FUNC × VERB_I+FUNC) ---")
p = plot_all_heads_grid('qk', vi_func_ids, vi_func_ids, vi_func_labels, vi_func_labels,
    f'QK circuits zoomed: VERB_I+FUNC (quote_content) — {MODEL_TAG}',
    'zoom_allheads_qk_quote.png')
print(f"Saved: {p}")

print("--- quote_content: all heads OV (VERB_I+FUNC → VERB_I+FUNC) ---")
p = plot_all_heads_grid('ov', vi_func_ids, vi_func_ids, vi_func_labels, vi_func_labels,
    f'OV circuits zoomed: VERB_I+FUNC (quote_content) — {MODEL_TAG}',
    'zoom_allheads_ov_quote.png')
print(f"Saved: {p}")


# %%
# ========================================================================
# PHASE 4: LOGIT ATTRIBUTION (exact decomposition)
# ========================================================================
print("\n" + "=" * 70)
print("PHASE 4: LOGIT ATTRIBUTION")
print("=" * 70)


def forward_with_cache(model, x):
    """Forward pass capturing per-head outputs and attention patterns.

    Returns logits and a cache dict with:
        embed: (B, T, d) embedding output
        head_outputs: list of 8 tensors (B, T, d), one per head across both layers
        patterns: list of 2 tensors (B, H, T, T), one per layer
    """
    B, T = x.shape
    n_head = model.config.n_head
    d_head = model.config.n_embd // n_head

    emb = model.embed(x)
    head_outputs = []
    patterns = []

    residual = emb
    for layer_idx, block in enumerate(model.blocks):
        attn = block.attn

        q1 = attn.q1(residual).view(B, T, n_head, d_head)
        k1 = attn.k1(residual).view(B, T, n_head, d_head)
        q2 = attn.q2(residual).view(B, T, n_head, d_head)
        k2 = attn.k2(residual).view(B, T, n_head, d_head)
        v = attn.v(residual).view(B, T, n_head, d_head)

        cos, sin = attn.rotary(q1)
        q1 = apply_rotary_emb(q1, cos, sin)
        k1 = apply_rotary_emb(k1, cos, sin)
        q2 = apply_rotary_emb(q2, cos, sin)
        k2 = apply_rotary_emb(k2, cos, sin)

        D = d_head
        scores1 = einsum(q1, k1, "b sq h d, b sk h d -> b h sq sk")
        scores2 = einsum(q2, k2, "b sq h d, b sk h d -> b h sq sk")

        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        pattern = (scores1 / D) * (scores2 / D)
        pattern = pattern.masked_fill(~causal_mask, 0.0)
        patterns.append(pattern.detach())

        # Per-head value-weighted outputs
        y = einsum(pattern, v, "b h sq sk, b sk h d -> b h sq d")

        # Per-head output through W_O slices
        for h in range(n_head):
            sl = slice(h * d_head, (h + 1) * d_head)
            W_O_h = attn.out.weight[:, sl]  # (d_model, d_head)
            head_out = y[:, h] @ W_O_h.T  # (B, T, d_model)
            head_outputs.append(head_out.detach())

        # Update residual
        y_full = y.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn.out(y_full)
        residual = residual + attn_out

    logits = model.lm_head(residual)
    return logits, {
        'embed': emb.detach(),
        'head_outputs': head_outputs,
        'patterns': patterns,
    }


def compute_logit_attribution(model, gen, n_batches=EVAL_BATCHES, batch_size=EVAL_BS, seq_len=EVAL_SEQ):
    """At each rule-fire position, compute each component's contribution to correct-token logit.

    Returns dict: rule -> array of shape (n_components,) where components are
    [direct, L0H0, L0H1, L0H2, L0H3, L1H0, L1H1, L1H2, L1H3].
    """
    eval_gen = LanguageV3(seed=999)
    n_components = 1 + N_LAYER * N_HEAD  # direct + heads
    rule_attribs = defaultdict(lambda: np.zeros(n_components))
    rule_counts = defaultdict(int)

    with torch.no_grad():
        for _ in range(n_batches):
            x, y, labels, mask = make_batch(eval_gen, batch_size, seq_len, device=device)
            logits, cache = forward_with_cache(model, x)

            # Verify decomposition: logits should equal sum of components
            lm = model.lm_head
            direct_logits = lm(cache['embed'])  # (B, T, V)
            head_logits = [lm(ho) for ho in cache['head_outputs']]  # list of (B, T, V)

            B, T = y.shape
            for b in range(B):
                for t in range(T):
                    lab = labels[b][t] if t < len(labels[b]) else 'pad'
                    if lab in ('pad', 'pre_context'):
                        continue
                    target_tok = y[b, t].item()

                    # Contribution of each component to the correct-token logit
                    attrib = np.zeros(n_components)
                    attrib[0] = direct_logits[b, t, target_tok].item()
                    for i, hl in enumerate(head_logits):
                        attrib[1 + i] = hl[b, t, target_tok].item()

                    rule_attribs[lab] += attrib
                    rule_counts[lab] += 1

    # Average
    result = {}
    for rule in rule_attribs:
        if rule_counts[rule] > 0:
            result[rule] = rule_attribs[rule] / rule_counts[rule]
    return result


# Verify exact decomposition on a single batch
print("Verifying exact logit decomposition...")
with torch.no_grad():
    x_test, y_test, _, _ = make_batch(LanguageV3(seed=123), 4, 65, device=device)
    logits_test, cache_test = forward_with_cache(model, x_test)
    lm = model.lm_head
    reconstructed = lm(cache_test['embed'])
    for ho in cache_test['head_outputs']:
        reconstructed = reconstructed + lm(ho)
    max_err = (logits_test - reconstructed).abs().max().item()
    print(f"  Max reconstruction error: {max_err:.2e} (should be ~0)")
    assert max_err < 1e-3, f"Decomposition error too large: {max_err}"

# Compute attribution
print("\nComputing logit attribution across eval data...")
attribution = compute_logit_attribution(model, gen)
component_names = ['direct'] + HEAD_LABELS

# Compute percentage attribution (% of total logit for correct token)
attrib_pct = {}
for rule, a in attribution.items():
    total = a.sum()
    if abs(total) > 0.01:
        attrib_pct[rule] = a / total * 100
    else:
        attrib_pct[rule] = np.zeros_like(a)

# Print attribution table with both raw and %
print(f"\n{'Rule':<20s} " + " ".join(f"{c:>7s}" for c in component_names) + "   total")
print("-" * (20 + 8 * len(component_names) + 10))
for rule in sorted(attribution, key=lambda r: baseline_losses.get(r, 99)):
    a = attribution[rule]
    total = a.sum()
    vals = " ".join(f"{v:>7.3f}" for v in a)
    print(f"{rule:<20s} {vals}  {total:>7.3f}")

print(f"\n{'Rule':<20s} " + " ".join(f"{c:>7s}" for c in component_names))
print("-" * (20 + 8 * len(component_names)))
for rule in sorted(attribution, key=lambda r: baseline_losses.get(r, 99)):
    p = attrib_pct[rule]
    vals = " ".join(f"{v:>6.1f}%" for v in p)
    print(f"{rule:<20s} {vals}")

# Attribution specialization summary
print(f"\n{'Rule':<20s} {'Top comp':>10s} {'Top%':>5s} {'Top2%':>6s}  Distribution (>10%)")
print("-" * 80)
for rule in sorted(attribution, key=lambda r: baseline_losses.get(r, 99)):
    p = attrib_pct[rule]
    sorted_idx = np.argsort(np.abs(p))[::-1]
    top1 = p[sorted_idx[0]]
    top2 = p[sorted_idx[0]] + p[sorted_idx[1]]

    dist_parts = []
    for idx in sorted_idx:
        if abs(p[idx]) > 10:
            dist_parts.append(f"{component_names[idx]}:{p[idx]:.0f}%")

    print(f"{rule:<20s} {component_names[sorted_idx[0]]:>10s} {top1:>4.0f}% {top2:>5.0f}%  {', '.join(dist_parts)}")

# %%
# --- Plot: percentage attribution horizontal stacked bars ---
rules_to_plot = sorted(attribution.keys(), key=lambda r: baseline_losses.get(r, 99))
n_rules = len(rules_to_plot)

fig, ax = plt.subplots(figsize=(12, max(5, n_rules * 0.45)))
y_pos = np.arange(n_rules)
colors = plt.cm.Set3(np.linspace(0, 1, len(component_names)))

lefts_pos = np.zeros(n_rules)
lefts_neg = np.zeros(n_rules)

for comp_idx, comp_name in enumerate(component_names):
    vals = np.array([attrib_pct[r][comp_idx] for r in rules_to_plot])
    pos_vals = np.maximum(vals, 0)
    neg_vals = np.minimum(vals, 0)

    bars = ax.barh(y_pos, pos_vals, left=lefts_pos, height=0.7,
                   label=comp_name, color=colors[comp_idx])
    ax.barh(y_pos, neg_vals, left=lefts_neg, height=0.7,
            color=colors[comp_idx])

    # Label bars > 15%
    for i, (pv, lp) in enumerate(zip(pos_vals, lefts_pos)):
        if pv > 15:
            ax.text(lp + pv / 2, i, f'{pv:.0f}%', ha='center', va='center', fontsize=7)

    lefts_pos += pos_vals
    lefts_neg += neg_vals

ax.set_yticks(y_pos)
ax.set_yticklabels(rules_to_plot, fontsize=9)
ax.set_xlabel('% of correct-token logit')
ax.set_title(f'Logit Attribution: % Contribution per Component — {MODEL_TAG}')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-10, 110)
ax.grid(True, alpha=0.2, axis='x')
ax.invert_yaxis()
fig.tight_layout()
path = os.path.join(FIG_DIR, 'logit_attribution_pct.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {path}")

# Also save the raw logit version
fig, ax = plt.subplots(figsize=(max(10, n_rules * 0.9), 6))
x_pos = np.arange(n_rules)

bottoms_pos = np.zeros(n_rules)
bottoms_neg = np.zeros(n_rules)

for comp_idx, comp_name in enumerate(component_names):
    vals = np.array([attribution[r][comp_idx] for r in rules_to_plot])
    pos_vals = np.maximum(vals, 0)
    neg_vals = np.minimum(vals, 0)

    ax.bar(x_pos, pos_vals, bottom=bottoms_pos, width=0.7,
           label=comp_name, color=colors[comp_idx])
    ax.bar(x_pos, neg_vals, bottom=bottoms_neg, width=0.7,
           color=colors[comp_idx])

    bottoms_pos += pos_vals
    bottoms_neg += neg_vals

ax.set_xticks(x_pos)
ax.set_xticklabels(rules_to_plot, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Contribution to correct-token logit (raw)')
ax.set_title(f'Logit Attribution by Component (raw logits) — {MODEL_TAG}')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
path = os.path.join(FIG_DIR, 'logit_attribution_raw.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {path}")


# %%
# ========================================================================
# PHASE 3: AUTOMATED CIRCUIT-RULE TESTS
# ========================================================================
print("\n" + "=" * 70)
print("PHASE 3: AUTOMATED CIRCUIT-RULE TESTS")
print("=" * 70)


@dataclass
class TestResult:
    test_name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    details: str = ''


all_test_results: list[TestResult] = []


def test_place_bigrams(direct, gen):
    """Check that direct path encodes place → func bigrams."""
    results = []
    direct_np = direct.cpu().numpy()
    for place, func_tok in gen.place_bigrams.items():
        place_id = gen.token2id[place]
        func_id = gen.token2id[func_tok]
        row = direct_np[place_id]
        rank = (row >= row[func_id]).sum()
        results.append(TestResult(
            test_name=f'place_bigram({place}→{func_tok})',
            passed=rank <= 3,
            metric_name='rank',
            metric_value=rank,
            threshold=3,
            details=f'direct[{place},{func_tok}]={row[func_id]:.3f}',
        ))
    return results


def test_noun_bigrams(direct, gen):
    """Check that direct path encodes noun → verb_t probabilistic bigrams."""
    results = []
    direct_np = direct.cpu().numpy()
    for noun, dist in gen.noun_bigram_dists.items():
        noun_id = gen.token2id[noun]
        row = direct_np[noun_id]
        expected_verbs = [v for v, p in dist]
        expected_ids = [gen.token2id[v] for v in expected_verbs]
        top_ids = np.argsort(row)[::-1]

        # Check if top verb matches highest-prob verb
        verb_ranks = [np.where(top_ids == vid)[0][0] + 1 for vid in expected_ids]
        best_verb_rank = verb_ranks[0]  # 70% verb should be ranked highest among the 3

        # Check ordering: 70% verb should have higher logit than 20% and 10%
        vals = [row[vid] for vid in expected_ids]
        order_correct = vals[0] >= vals[1] >= vals[2]

        results.append(TestResult(
            test_name=f'noun_bigram({noun})',
            passed=best_verb_rank <= 5 and order_correct,
            metric_name='top_verb_rank',
            metric_value=best_verb_rank,
            threshold=5,
            details=f'order_correct={order_correct}, ranks={verb_ranks}',
        ))
    return results


def test_tok_trigrams(circuits, delta_losses, gen):
    """Check that ablation-identified heads have relevant QK/OV structure for trigrams."""
    results = []
    for (prev2, prev), output in gen.tok_trigrams.items():
        # Find most responsible head via ablation
        rule = 'tok_trigram'
        best_head = max(HEAD_LABELS, key=lambda h: delta_losses[h].get(rule, 0))
        best_delta = delta_losses[best_head].get(rule, 0)

        layer = int(best_head[1])
        head = int(best_head[3])

        prev_id = gen.token2id[prev]
        prev2_id = gen.token2id[prev2]
        output_id = gen.token2id[output]

        # Check QK: does this head attend from prev to prev2?
        qk = circuits[f'L{layer}_qk'][head].cpu().numpy()
        qk_val = qk[prev_id, prev2_id]
        qk_row = qk[prev_id]
        qk_rank = (qk_row >= qk_val).sum()

        # Check OV: when attending to prev2, does it promote output?
        ov = circuits[f'L{layer}_ov'][head].cpu().numpy()
        ov_val = ov[prev2_id, output_id]
        ov_row = ov[prev2_id]
        ov_rank = (ov_row >= ov_val).sum()

        results.append(TestResult(
            test_name=f'tok_trigram({prev2},{prev}→{output})',
            passed=qk_rank <= config.vocab_size // 4 and ov_rank <= config.vocab_size // 4,
            metric_name='qk_rank,ov_rank',
            metric_value=qk_rank,
            threshold=config.vocab_size // 4,
            details=f'head={best_head}(Δ={best_delta:.3f}), qk_rank={qk_rank}, ov_rank={ov_rank}',
        ))
    return results


def test_skip_bigrams(circuits, delta_losses, gen):
    """Check that responsible head's QK attends from trigger to anchor."""
    results = []
    rule = 'skip_bigram'
    best_head = max(HEAD_LABELS, key=lambda h: delta_losses[h].get(rule, 0))
    layer = int(best_head[1])
    head = int(best_head[3])

    for (anchor, trigger), output in gen.skip_bigrams.items():
        anchor_id = gen.token2id[anchor]
        trigger_id = gen.token2id[trigger]

        qk = circuits[f'L{layer}_qk'][head].cpu().numpy()
        qk_val = qk[trigger_id, anchor_id]
        qk_row = qk[trigger_id]
        qk_rank = (qk_row >= qk_val).sum()

        results.append(TestResult(
            test_name=f'skip_bigram({anchor}...{trigger}→{output})',
            passed=qk_rank <= 10,
            metric_name='qk_rank',
            metric_value=qk_rank,
            threshold=10,
            details=f'head={best_head}, qk[{trigger},{anchor}]={qk_val:.3f}',
        ))
    return results


def compute_composition_scores(model):
    """Compute V-composition and K-composition matrices between L0 and L1 heads.

    Returns dict with keys like 'V_comp', 'K1_comp', 'K2_comp', 'Q1_comp', 'Q2_comp',
    each a (N_HEAD, N_HEAD) matrix of Frobenius norms.
    """
    L0 = model.blocks[0].attn
    L1 = model.blocks[1].attn

    comp = {}
    for name, L1_weight_name in [
        ('V_comp', 'v'),
        ('K1_comp', 'k1'), ('K2_comp', 'k2'),
        ('Q1_comp', 'q1'), ('Q2_comp', 'q2'),
    ]:
        mat = np.zeros((N_HEAD, N_HEAD))
        for h0 in range(N_HEAD):
            sl0 = slice(h0 * D_HEAD, (h0 + 1) * D_HEAD)
            W_O_L0 = L0.out.weight[:, sl0].detach().float()  # (d, dh)

            for h1 in range(N_HEAD):
                sl1 = slice(h1 * D_HEAD, (h1 + 1) * D_HEAD)
                W_L1 = getattr(L1, L1_weight_name).weight[sl1].detach().float()  # (dh, d)
                composition = W_L1 @ W_O_L0  # (dh, dh)
                mat[h0, h1] = composition.norm().item()

        comp[name] = mat
    return comp


def test_induction(comp_scores, delta_losses):
    """Check that V-composition between some L0-L1 pair is dominant, consistent with induction."""
    results = []
    rule = 'induction'

    # Find the head pair with strongest V-composition
    v_comp = comp_scores['V_comp']
    max_val = v_comp.max()
    mean_val = v_comp.mean()
    max_idx = np.unravel_index(v_comp.argmax(), v_comp.shape)
    ratio = max_val / mean_val

    # Also check which L1 head is most important for induction via ablation
    l1_heads = [f'L1H{h}' for h in range(N_HEAD)]
    best_l1 = max(l1_heads, key=lambda h: delta_losses[h].get(rule, 0))
    best_l1_delta = delta_losses[best_l1].get(rule, 0)

    results.append(TestResult(
        test_name='induction_v_composition',
        passed=ratio > 1.5,
        metric_name='max/mean V-comp ratio',
        metric_value=ratio,
        threshold=1.5,
        details=f'strongest pair: L0H{max_idx[0]}→L1H{max_idx[1]} '
                f'(val={max_val:.3f}, mean={mean_val:.3f}), '
                f'ablation: {best_l1}(Δ={best_l1_delta:.3f})',
    ))
    return results


def test_category_structure(gen):
    """Check that embeddings cluster by category (silhouette-like score)."""
    E = model.embed.weight.detach().float().cpu().numpy()

    # Compute cosine similarities
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_normed = E / (norms + 1e-8)

    intra_sims = []
    inter_sims = []
    for i in range(len(CAT_NAMES)):
        start_i, end_i = CAT_BOUNDS[i], CAT_BOUNDS[i + 1]
        embs_i = E_normed[start_i:end_i]
        if len(embs_i) < 2:
            continue

        # Intra-category similarity
        sim_mat = embs_i @ embs_i.T
        n = len(embs_i)
        for a in range(n):
            for b in range(a + 1, n):
                intra_sims.append(sim_mat[a, b])

        # Inter-category similarity
        for j in range(len(CAT_NAMES)):
            if i == j:
                continue
            start_j, end_j = CAT_BOUNDS[j], CAT_BOUNDS[j + 1]
            embs_j = E_normed[start_j:end_j]
            cross_sim = embs_i @ embs_j.T
            inter_sims.extend(cross_sim.flatten().tolist())

    mean_intra = np.mean(intra_sims)
    mean_inter = np.mean(inter_sims)
    gap = mean_intra - mean_inter

    return [TestResult(
        test_name='category_embedding_structure',
        passed=gap > 0.1,
        metric_name='intra-inter cosine gap',
        metric_value=gap,
        threshold=0.1,
        details=f'intra={mean_intra:.3f}, inter={mean_inter:.3f}',
    )]


def test_cat_bigram_correlation(gen):
    """Check that category-averaged direct path correlates with POS transition matrix."""
    cat_direct = compute_category_circuit(circuits['direct'])

    # Compare with transition matrix (only the 7 POS categories)
    # cat_direct rows/cols: NOUN, PLACE, VERB_T, VERB_I, ADJ, FUNC, STRUCT
    # transition matrix: NOUN, PLACE, VERB_T, VERB_I, ADJ, FUNC, PUNCT
    n_pos = len(CAT_ORDER)  # 7
    direct_sub = cat_direct[:n_pos, :n_pos]
    trans = TRANSITION_MATRIX

    # Flatten and correlate
    d_flat = direct_sub.flatten()
    t_flat = trans.flatten()
    corr = np.corrcoef(d_flat, t_flat)[0, 1]

    return [TestResult(
        test_name='cat_bigram_transition_correlation',
        passed=corr > 0.3,
        metric_name='Pearson r',
        metric_value=corr,
        threshold=0.3,
        details=f'correlation between category-avg direct path and POS transitions',
    )]


# Run all tests
print("\nRunning automated circuit tests...\n")

all_test_results.extend(test_place_bigrams(circuits['direct'], gen))
all_test_results.extend(test_noun_bigrams(circuits['direct'], gen))
all_test_results.extend(test_tok_trigrams(circuits, delta_losses, gen))
all_test_results.extend(test_skip_bigrams(circuits, delta_losses, gen))

comp_scores = compute_composition_scores(model)
all_test_results.extend(test_induction(comp_scores, delta_losses))
all_test_results.extend(test_category_structure(gen))
all_test_results.extend(test_cat_bigram_correlation(gen))

# Print results
n_pass = sum(1 for t in all_test_results if t.passed)
n_total = len(all_test_results)
print(f"Results: {n_pass}/{n_total} tests passed ({100*n_pass/n_total:.0f}%)\n")
for t in all_test_results:
    status = "PASS" if t.passed else "FAIL"
    print(f"  [{status}] {t.test_name}")
    print(f"         {t.metric_name}={t.metric_value:.3f} (threshold={t.threshold})")
    if t.details:
        print(f"         {t.details}")

# %%
# --- Composition score heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, ['V_comp', 'K1_comp', 'Q1_comp']):
    mat = comp_scores[name]
    vmax = mat.max()
    im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='equal')
    ax.set_xticks(range(N_HEAD))
    ax.set_xticklabels([f'L1H{h}' for h in range(N_HEAD)])
    ax.set_yticks(range(N_HEAD))
    ax.set_yticklabels([f'L0H{h}' for h in range(N_HEAD)])
    ax.set_title(name.replace('_', ' '))
    fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(N_HEAD):
        for j in range(N_HEAD):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=8)
fig.suptitle(f'L0→L1 Composition Scores — {MODEL_TAG}', fontsize=11)
fig.tight_layout()
path = os.path.join(FIG_DIR, 'composition_scores.png')
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"\nSaved: {path}")


# %%
# ========================================================================
# PHASE 5: ATTENTION PATTERN ANALYSIS
# ========================================================================
print("\n" + "=" * 70)
print("PHASE 5: ATTENTION PATTERN ANALYSIS")
print("=" * 70)


def analyze_attention_patterns(model, gen, n_batches=EVAL_BATCHES, batch_size=EVAL_BS, seq_len=EVAL_SEQ):
    """Compute mean attention at each relative position, per head per rule.

    Returns:
        patterns_by_rule: dict rule -> (n_total_heads, max_rel_pos) mean attention
    """
    eval_gen = LanguageV3(seed=999)
    n_total_heads = N_LAYER * N_HEAD
    max_rel = min(32, seq_len)  # look back up to 32 positions

    rule_attn_sums = defaultdict(lambda: np.zeros((n_total_heads, max_rel)))
    rule_counts = defaultdict(int)

    with torch.no_grad():
        for _ in range(n_batches):
            x, y, labels, mask = make_batch(eval_gen, batch_size, seq_len, device=device)
            _, cache = forward_with_cache(model, x)

            B, T = y.shape
            for b in range(B):
                for t in range(T):
                    lab = labels[b][t] if t < len(labels[b]) else 'pad'
                    if lab in ('pad', 'pre_context'):
                        continue

                    for layer_idx in range(N_LAYER):
                        pattern = cache['patterns'][layer_idx]  # (B, H, T, T)
                        for h in range(N_HEAD):
                            head_idx = layer_idx * N_HEAD + h
                            attn_row = pattern[b, h, t, :t+1].cpu().numpy()
                            # Convert to relative positions
                            for rel in range(min(len(attn_row), max_rel)):
                                src_pos = t - rel
                                if src_pos >= 0:
                                    rule_attn_sums[lab][head_idx, rel] += attn_row[src_pos]

                    rule_counts[lab] += 1

    # Normalize
    result = {}
    for rule in rule_attn_sums:
        if rule_counts[rule] > 0:
            result[rule] = rule_attn_sums[rule] / rule_counts[rule]
    return result


print("Computing attention patterns at rule-fire positions...")
attn_patterns = analyze_attention_patterns(model, gen)

# Plot attention patterns for key rules (RdBu_r: red=positive attn, blue=negative, white=0)
# Reminder: bilinear attention is (Q1·K1/D)*(Q2·K2/D), NOT softmax-normalized.
# Values can be >1 or <0. Negative = head actively suppresses that position.
key_rules = ['place_bigram', 'noun_bigram', 'tok_trigram', 'skip_bigram',
             'induction', 'cat_trigram', 'cat_bigram', 'skip_trigram']
key_rules = [r for r in key_rules if r in attn_patterns]

N_REL_SHOW = 20  # show first 20 relative positions

fig, axes = plt.subplots(2, 4, figsize=(22, 8))
axes = axes.flatten()
for idx, rule in enumerate(key_rules):
    if idx >= len(axes):
        break
    ax = axes[idx]
    data = attn_patterns[rule][:, :N_REL_SHOW]
    vmax = max(abs(data.min()), abs(data.max()), 0.01)
    im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xlabel('Relative position (0=self, 1=prev, ...)')
    ax.set_xticks(range(0, N_REL_SHOW, 2))
    ax.set_yticks(range(N_LAYER * N_HEAD))
    ax.set_yticklabels(HEAD_LABELS, fontsize=8)
    ax.set_title(rule, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7)

for idx in range(len(key_rules), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(
    f'Mean Attention Weight by Relative Position at Rule-Fire — {MODEL_TAG}\n'
    '(red=positive attention, blue=negative/suppression, white=zero)',
    fontsize=11)
fig.tight_layout()
path = os.path.join(FIG_DIR, 'attention_patterns_by_rule.png')
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# %%
# --- Detailed walkthrough of attention patterns ---
# Context: at position t, the model sees x[0..t] and predicts y[t].
#   rel=0 means attending to x[t] (self = last input token before prediction)
#   rel=1 means attending to x[t-1] (one token back)
#   rel=k means attending to x[t-k]
#
# For each rule type, which position matters:
#   place_bigram:  x[t]=place → y[t]=func.       Model needs rel=0 (self).
#   noun_bigram:   x[t]=noun → y[t]=verb.         Model needs rel=0 (self).
#   tok_trigram:   x[t-1]=prev2, x[t]=prev → y[t]. Model needs rel=0 + rel=1.
#   cat_trigram:   x[t-1]=cat1, x[t]=cat2 → y[t].  Model needs rel=0 + rel=1.
#   skip_bigram:   x[t-k]=anchor, x[t]=trigger → y[t]. Model needs rel=0 + rel=k (k>=2).
#   induction:     x[t]=repeated noun, need to find first occurrence at x[t-j]
#                  and read x[t-j+1] (follower). Model needs variable rel.
#   skip_trigram:  bracket close. Model must attend back to bracket opener.

print("\n" + "-" * 80)
print("ATTENTION PATTERN WALKTHROUGH")
print("-" * 80)
print("""
Position key:  rel=0 is self (x[t]), rel=1 is one-back (x[t-1]), etc.
Bilinear attention is unnormalized — values can be >1 or <0.
""")

for rule in key_rules:
    data = attn_patterns[rule]
    print(f"\n  === {rule} ===")

    # Show the full distribution for each active head
    for head_idx, head_name in enumerate(HEAD_LABELS):
        row = data[head_idx, :N_REL_SHOW]
        peak = np.argmax(np.abs(row))
        peak_val = row[peak]
        if abs(peak_val) < 0.01:
            continue

        # Show top-3 positions by absolute value
        top3_idx = np.argsort(np.abs(row))[::-1][:3]
        top3 = [(int(i), row[i]) for i in top3_idx if abs(row[i]) > 0.005]
        top3_str = ", ".join(f"rel={i}:{v:+.3f}" for i, v in top3)
        print(f"    {head_name}: {top3_str}")

    # Interpretation
    explanations = {
        'place_bigram': (
            "  Interpretation: L1H1/L1H2 strongly self-attend (rel=0). The place token is at\n"
            "  rel=0 — these heads read it and write the associated FUNC word. The direct path\n"
            "  (embed→unembed) handles the token-specific mapping, heads amplify it."
        ),
        'noun_bigram': (
            "  Interpretation: L0H0 and L1H0 self-attend (rel=0). The noun is at rel=0.\n"
            "  Direct path contributes 62% of the logit (it encodes the bigram distribution).\n"
            "  L1H0 adds 29% by also reading the noun embedding. No need to look back."
        ),
        'tok_trigram': (
            "  Interpretation: L0H0 attends strongly to rel=1 (the prev2 token, e.g. 'the').\n"
            "  It writes prev2 info into the residual stream. L1H1/L1H2 self-attend (rel=0),\n"
            "  reading the enriched residual that now contains both prev (from embedding) and\n"
            "  prev2 (from L0H0). This is the classic 'L0 gathers context, L1 reads it' pattern."
        ),
        'cat_trigram': (
            "  Interpretation: Same mechanism as tok_trigram. L0H0 looks back to rel=1\n"
            "  (the FUNC token in 'FUNC NOUN → VERB_I'). L1H0 and L1H3 self-attend to\n"
            "  combine the category info. L0H0 handles 89% of the ablation impact."
        ),
        'skip_bigram': (
            "  Interpretation: L1H1 has strong attention at rel=1. But the anchor is at rel>=2.\n"
            "  Possible explanations: (1) L0 heads write anchor identity into the residual at\n"
            "  intermediate positions, so L1H1 doesn't need to directly attend to the anchor.\n"
            "  (2) The trigger boost mechanism places triggers near anchors (within 4 pos),\n"
            "  so the effective distance is short and gets averaged. Check rel=2..6 for spread."
        ),
        'induction': (
            "  Interpretation: L0H0 self-attends (rel=0) to read the repeated noun.\n"
            "  L1H0 self-attends (rel=0) to read L0's output. The actual 'find first occurrence'\n"
            "  attention is spread across variable relative positions (depends on where the first\n"
            "  noun appeared), so no single peak. L0H3 (51% ablation) + L1H1 (26%) carry this."
        ),
        'cat_bigram': (
            "  Interpretation: Weak attention across all heads — this is the fallback rule.\n"
            "  The direct path (39%) and L1H3 (41%) handle most of it. Category bigram doesn't\n"
            "  need long-range attention, just the current token's category."
        ),
        'skip_trigram': (
            "  Interpretation: Bracket close (e.g., ')' or ']'). The model must attend back to\n"
            "  the bracket opener to predict the close token. L1H1 and L1H3 show attention at\n"
            "  rel=4 (typical bracket span of 2-4 content tokens + opener). L1H2 at rel=2-3.\n"
            "  This is the hardest rule (KL=0.60) because the distance to the opener varies."
        ),
    }
    if rule in explanations:
        print(explanations[rule])


# %%
# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Test results
n_pass = sum(1 for t in all_test_results if t.passed)
n_total = len(all_test_results)
print(f"\nAutomated tests: {n_pass}/{n_total} passed ({100*n_pass/n_total:.0f}%)")

# Combined view: ablation + attribution for each rule
print(f"\n{'Rule':<18s} │ {'KL':>5s} │ {'Ablation (% of impact)':^44s} │ {'Attribution (% of logit)':^44s} │ Type")
print("─" * 120)
for rule in sorted(baseline_losses, key=baseline_losses.get):
    kl = baseline_losses[rule] - h_true.get(rule, 0)

    # Ablation distribution
    j = rule_names.index(rule)
    abl_pcts = pct_matrix[:, j]
    abl_sorted = np.argsort(abl_pcts)[::-1]
    abl_parts = []
    for idx in abl_sorted:
        if abl_pcts[idx] > 5:
            abl_parts.append(f"{head_names[idx]}:{abl_pcts[idx]:.0f}%")
    abl_str = ", ".join(abl_parts[:4]) if abl_parts else "—"
    abl_top1 = abl_pcts[abl_sorted[0]]

    # Attribution distribution
    if rule in attrib_pct:
        att_pcts = attrib_pct[rule]
        att_sorted = np.argsort(np.abs(att_pcts))[::-1]
        att_parts = []
        for idx in att_sorted:
            if abs(att_pcts[idx]) > 10:
                att_parts.append(f"{component_names[idx]}:{att_pcts[idx]:.0f}%")
        att_str = ", ".join(att_parts[:4]) if att_parts else "—"
        att_top1 = abs(att_pcts[att_sorted[0]])
    else:
        att_str = "—"
        att_top1 = 0

    # Classify specialization
    if abl_top1 > 60 or att_top1 > 50:
        spec = "FOCUSED"
    elif abl_top1 < 30:
        spec = "SHARED"
    else:
        spec = "MIXED"

    print(f"{rule:<18s} │ {kl:>5.2f} │ {abl_str:<44s} │ {att_str:<44s} │ {spec}")

# Figures saved
print(f"\nAll figures saved to: {FIG_DIR}/")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

# %%

# %%
"""
Evaluate trained BilinearGPT models on isolated rule datasets.
Loads a model checkpoint and measures per-rule loss on each rule type.
"""
import sys
import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BilinearGPT, Config
from generator import (
    LanguageGenerator, VOCAB_SIZE, ID2TOKEN, TOKEN2ID, VOCAB,
    BIGRAM_RULES, TRIGRAM_RULES, SKIP_BIGRAM_RULES,
    INDUCTION_RULES, BRACKET_RULES, DEPTH1_RULES, ALL_RULES,
    DEPTH1_MIXING, DEFAULT_MIXING, make_isolated_test_set,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# === CONFIG: set model path here ===
MODEL_TAG = 'all_2L_4H_d64_10000steps'
MODEL_PATH = os.path.join(ROOT_DIR, 'models', f'bilinear_gpt_{MODEL_TAG}.pt')
CONFIG_PATH = os.path.join(ROOT_DIR, 'configs', f'bilinear_gpt_{MODEL_TAG}.json')

# %%
# === Load model ===
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
print(f"Loaded {MODEL_TAG}: {model.count_params():,} params, {config.n_layer}L {config.n_head}H d={config.n_embd}")

# %%
# === QK / OV Circuit Analysis ===
# Bilinear QK circuit: (E @ W_Q1^T @ W_K1 @ E^T) * (E @ W_Q2^T @ W_K2 @ E^T) / D^2
# OV circuit: E @ W_V^T @ W_O^T @ U^T
# Direct path: E @ U^T

def extract_circuits(model, layer=0):
    """Extract per-head QK, OV circuits and direct path from a BilinearGPT model.

    Returns:
        qk_circuits: list of (V, V) tensors, one per head (bilinear QK combined)
        qk1_circuits: list of (V, V) tensors (first QK pair only)
        qk2_circuits: list of (V, V) tensors (second QK pair only)
        ov_circuits: list of (V, V) tensors, one per head
        direct: (V, V) tensor (embed @ unembed^T)
    """
    attn = model.blocks[layer].attn
    n_head = attn.n_head
    dh = attn.head_dim
    D = dh  # normalization factor from forward pass

    E = model.embed.weight.detach().float()   # (V, d)
    U = model.lm_head.weight.detach().float()  # (V, d)

    qk_circuits = []
    qk1_circuits = []
    qk2_circuits = []
    ov_circuits = []

    for h in range(n_head):
        sl = slice(h * dh, (h + 1) * dh)

        # QK circuits (ignoring RoPE — content-only)
        W_Q1 = attn.q1.weight[sl].detach().float()  # (dh, d)
        W_K1 = attn.k1.weight[sl].detach().float()
        W_Q2 = attn.q2.weight[sl].detach().float()
        W_K2 = attn.k2.weight[sl].detach().float()

        QK1 = E @ W_Q1.T @ W_K1 @ E.T  # (V, V)
        QK2 = E @ W_Q2.T @ W_K2 @ E.T  # (V, V)
        QK_combined = (QK1 * QK2) / (D ** 2)

        qk1_circuits.append(QK1)
        qk2_circuits.append(QK2)
        qk_circuits.append(QK_combined)

        # OV circuit
        W_V = attn.v.weight[sl].detach().float()       # (dh, d)
        W_O = attn.out.weight[:, sl].detach().float()   # (d, dh)
        OV = E @ W_V.T @ W_O.T @ U.T  # (V, V)
        ov_circuits.append(OV)

    # Direct path: embed -> unembed (skip connection)
    direct = E @ U.T  # (V, V)

    return qk_circuits, qk1_circuits, qk2_circuits, ov_circuits, direct


# Extract circuits
qk_circuits, qk1_circuits, qk2_circuits, ov_circuits, direct_path = extract_circuits(model)
n_head = config.n_head
print(f"Extracted circuits for {n_head} heads, vocab={config.vocab_size}")

# %%
# === Visualize circuits ===
# Token labels for axes
token_labels = [ID2TOKEN[i] for i in range(config.vocab_size)]

def plot_circuit_matrix(mat, title, ax, token_labels, cmap='RdBu_r', symmetric=True):
    """Plot a V×V circuit matrix as a heatmap."""
    data = mat.cpu().numpy()
    if symmetric:
        vmax = max(abs(data.min()), abs(data.max()))
        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
    else:
        im = ax.imshow(data, cmap=cmap, aspect='equal')
    ax.set_title(title, fontsize=10)
    if len(token_labels) <= 30:
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(token_labels)))
        ax.set_yticklabels(token_labels, fontsize=7)
    ax.set_xlabel('Key / Source token')
    ax.set_ylabel('Query / Dest token')
    return im

# --- Direct path ---
fig, ax = plt.subplots(figsize=(8, 7))
im = plot_circuit_matrix(direct_path, 'Direct path: E @ U^T', ax, token_labels)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
ATTN_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', MODEL_TAG)
os.makedirs(ATTN_FIG_DIR, exist_ok=True)
plot_path = os.path.join(ATTN_FIG_DIR, f'circuit_direct_{MODEL_TAG}.png')
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot_path}")

# --- Per-head QK circuits (combined bilinear) ---
fig, axes = plt.subplots(1, n_head, figsize=(5 * n_head, 5))
if n_head == 1:
    axes = [axes]
for h in range(n_head):
    im = plot_circuit_matrix(qk_circuits[h], f'Head {h}: QK combined\n(QK1*QK2)/D²',
                             axes[h], token_labels)
    fig.colorbar(im, ax=axes[h], shrink=0.6)
fig.suptitle(f'Bilinear QK Circuits (content-only, no RoPE) — {MODEL_TAG}', fontsize=12)
fig.tight_layout()
plot_path = os.path.join(ATTN_FIG_DIR, f'circuit_qk_{MODEL_TAG}.png')
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot_path}")

# --- Per-head QK1 and QK2 separately ---
fig, axes = plt.subplots(2, n_head, figsize=(5 * n_head, 10))
if n_head == 1:
    axes = axes.reshape(2, 1)
for h in range(n_head):
    im1 = plot_circuit_matrix(qk1_circuits[h], f'Head {h}: QK1', axes[0, h], token_labels)
    fig.colorbar(im1, ax=axes[0, h], shrink=0.6)
    im2 = plot_circuit_matrix(qk2_circuits[h], f'Head {h}: QK2', axes[1, h], token_labels)
    fig.colorbar(im2, ax=axes[1, h], shrink=0.6)
fig.suptitle(f'Individual QK pairs (content-only) — {MODEL_TAG}', fontsize=12)
fig.tight_layout()
plot_path = os.path.join(ATTN_FIG_DIR, f'circuit_qk_pairs_{MODEL_TAG}.png')
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot_path}")

# --- Per-head OV circuits ---
fig, axes = plt.subplots(1, n_head, figsize=(5 * n_head, 5))
if n_head == 1:
    axes = [axes]
for h in range(n_head):
    im = plot_circuit_matrix(ov_circuits[h], f'Head {h}: OV\nE @ Wv^T @ Wo^T @ U^T',
                             axes[h], token_labels)
    fig.colorbar(im, ax=axes[h], shrink=0.6)
fig.suptitle(f'OV Circuits — {MODEL_TAG}', fontsize=12)
fig.tight_layout()
plot_path = os.path.join(ATTN_FIG_DIR, f'circuit_ov_{MODEL_TAG}.png')
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved: {plot_path}")

# %%
# === Full attention circuit: QK * OV (what logits change when head attends?) ===
# For each head: the "full circuit" combines WHERE it attends (QK) with WHAT it writes (OV)
# Full_h[query_tok, output_tok] = sum_key QK_h[query, key] * OV_h[key, output]

print("\n" + "=" * 60)
print("Circuit statistics")
print("=" * 60)

for h in range(n_head):
    qk = qk_circuits[h]
    ov = ov_circuits[h]
    print(f"\nHead {h}:")
    print(f"  QK combined — mean: {qk.mean():.4f}, std: {qk.std():.4f}, "
          f"max: {qk.max():.4f}, min: {qk.min():.4f}")
    print(f"  QK1         — mean: {qk1_circuits[h].mean():.4f}, std: {qk1_circuits[h].std():.4f}")
    print(f"  QK2         — mean: {qk2_circuits[h].mean():.4f}, std: {qk2_circuits[h].std():.4f}")
    print(f"  OV          — mean: {ov.mean():.4f}, std: {ov.std():.4f}, "
          f"max: {ov.max():.4f}, min: {ov.min():.4f}")

print(f"\nDirect path — mean: {direct_path.mean():.4f}, std: {direct_path.std():.4f}")

# %%
# === Helper functions (global) ===

def eval_on_generator(gen, n_batches=50, batch_size=128, seq_len=64):
    """Evaluate model on a generator, return overall loss and per-rule losses."""
    all_losses = []
    rule_losses = defaultdict(list)

    with torch.no_grad():
        for _ in range(n_batches):
            all_tokens, all_labels = gen.sample_batch(batch_size, length=seq_len + 1)
            tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
            x, y = tokens[:, :-1], tokens[:, 1:]
            labels = [lab[1:] for lab in all_labels]

            logits, loss = model(x, targets=y)
            all_losses.append(loss.item())

            # Per-token CE
            B, T, V = logits.shape
            ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction='none').view(B, T)
            for b in range(B):
                for t in range(T):
                    lab = labels[b][t]
                    if lab.startswith('bracket'):
                        lab = 'bracket'
                    rule_losses[lab].append(ce[b, t].item())

    overall = np.mean(all_losses)
    per_rule = {k: np.mean(v) for k, v in rule_losses.items()}
    return overall, per_rule


def eval_isolated_rule(rule, n_seqs=500, seq_len=64):
    """Evaluate model on sequences generated by a single isolated rule.
    Returns (mean_loss_at_fire_positions, mean_loss_overall, n_fire_positions)."""
    sequences, labels, fire_positions = make_isolated_test_set(rule, n=n_seqs, length=seq_len + 1, seed=42)

    tokens = torch.tensor(sequences, dtype=torch.long, device=device)
    x, y = tokens[:, :-1], tokens[:, 1:]
    # fire_positions index into the original sequence; shift by -1 for targets
    # (fire at position i means token i was produced by the rule, so we measure
    # loss on predicting token i, which is target position i-1)

    all_fire_losses = []
    all_losses = []

    with torch.no_grad():
        # Process in chunks to avoid OOM
        chunk = 128
        for start in range(0, len(x), chunk):
            x_chunk = x[start:start + chunk]
            y_chunk = y[start:start + chunk]
            logits, loss = model(x_chunk, targets=y_chunk)
            B, T, V = logits.shape
            ce = F.cross_entropy(logits.reshape(-1, V), y_chunk.reshape(-1), reduction='none').view(B, T)

            for b_local in range(B):
                b_global = start + b_local
                all_losses.append(ce[b_local].mean().item())
                for pos in fire_positions[b_global]:
                    # Target position is pos-1 (predicting token at pos given context up to pos-1)
                    target_pos = pos - 1
                    if 0 <= target_pos < T:
                        all_fire_losses.append(ce[b_local, target_pos].item())

    return (
        np.mean(all_fire_losses) if all_fire_losses else float('nan'),
        np.mean(all_losses),
        len(all_fire_losses),
    )

# %%
# === Evaluate on mixed depth-1 data ===
print("=" * 60)
print("Mixed depth-1 evaluation")
print("=" * 60)

gen_mixed = LanguageGenerator(rules=ALL_RULES, mixing_weights=DEFAULT_MIXING, mode='mixed', seed=99)
mixed_loss, mixed_per_rule = eval_on_generator(gen_mixed)
print(f"\nOverall val loss: {mixed_loss:.4f}\n")
print(f"{'Rule':<25s} {'Loss':>8s}")
print("-" * 35)
for k, v in sorted(mixed_per_rule.items(), key=lambda x: x[1]):
    print(f"{k:<25s} {v:>8.4f}")

# %%
# === Evaluate each rule in isolation ===
print("\n" + "=" * 60)
print("Isolated rule evaluation")
print("=" * 60)

all_rules_flat = BIGRAM_RULES + TRIGRAM_RULES + SKIP_BIGRAM_RULES + INDUCTION_RULES + BRACKET_RULES
iso_results = {}

for rule in all_rules_flat:
    fire_loss, overall_loss, n_fires = eval_isolated_rule(rule)
    iso_results[rule.name] = {
        'fire_loss': fire_loss,
        'overall_loss': overall_loss,
        'n_fires': n_fires,
        'rule_class': rule.rule_class,
    }
    status = "LEARNED" if fire_loss < 0.5 else ("PARTIAL" if fire_loss < 2.0 else "FAILED")
    print(f"  {rule.name:<25s} fire_loss={fire_loss:>7.4f}  (n={n_fires:>4d})  [{status}]")

# %%
# === Summary table: learned vs not ===
print("\n" + "=" * 60)
print("Summary by rule class")
print("=" * 60)

by_class = defaultdict(list)
for name, res in iso_results.items():
    by_class[res['rule_class']].append((name, res['fire_loss']))

for cls in ['bigram', 'trigram', 'skip_bigram', 'induction', 'bracket']:
    if cls not in by_class:
        continue
    rules = by_class[cls]
    mean_loss = np.mean([l for _, l in rules])
    status = "LEARNED" if mean_loss < 0.5 else ("PARTIAL" if mean_loss < 2.0 else "FAILED")
    print(f"\n  {cls} (mean fire_loss={mean_loss:.4f}) [{status}]")
    for name, loss in sorted(rules, key=lambda x: x[1]):
        print(f"    {name:<25s} {loss:.4f}")

# %%
# === Plot: per-rule isolated fire loss ===
names = []
losses = []
colors = []
color_map = {
    'bigram': '#2196F3',
    'trigram': '#4CAF50',
    'skip_bigram': '#FF9800',
    'induction': '#F44336',
    'bracket': '#9C27B0',
}

for rule in all_rules_flat:
    r = iso_results[rule.name]
    names.append(rule.name)
    losses.append(r['fire_loss'])
    colors.append(color_map.get(r['rule_class'], '#999'))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(len(names)), losses, color=colors)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Loss at rule fire positions')
ax.set_title(f'Per-rule isolated evaluation: {MODEL_TAG}')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='learned threshold')
ax.invert_yaxis()
ax.legend()
ax.grid(True, alpha=0.2, axis='x')
fig.tight_layout()

plot_path = os.path.join(ATTN_FIG_DIR, f'rule_eval_{MODEL_TAG}.png')
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

# %%
# === Inspect: show model predictions on sample sequences ===
print("\n" + "=" * 60)
print("Sample predictions")
print("=" * 60)

def show_predictions(rule, n_seqs=3, seq_len=32):
    """Print sequences with model predictions at rule fire positions."""
    sequences, labels, fire_positions = make_isolated_test_set(rule, n=n_seqs, length=seq_len + 1, seed=123)
    tokens = torch.tensor(sequences, dtype=torch.long, device=device)
    x, y = tokens[:, :-1], tokens[:, 1:]

    with torch.no_grad():
        logits, _ = model(x, targets=y)
        probs = F.softmax(logits, dim=-1)

    for i in range(n_seqs):
        seq_str = [ID2TOKEN[t] for t in sequences[i][:seq_len]]
        print(f"\n  Sequence: {' '.join(seq_str)}")
        print(f"  Labels:   {' '.join(labels[i][:seq_len])}")
        for pos in fire_positions[i]:
            target_pos = pos - 1
            if target_pos < 0 or target_pos >= seq_len:
                continue
            true_tok = ID2TOKEN[y[i, target_pos].item()]
            pred_probs = probs[i, target_pos]
            top5_idx = pred_probs.topk(5).indices.tolist()
            top5 = [(ID2TOKEN[idx], pred_probs[idx].item()) for idx in top5_idx]
            top5_str = ', '.join(f'{t}:{p:.3f}' for t, p in top5)
            correct_prob = pred_probs[y[i, target_pos].item()].item()
            print(f"    pos {pos}: true={true_tok} (p={correct_prob:.4f}) | top5: {top5_str}")


# Show a few examples from each rule class
for rule in BIGRAM_RULES[:2]:
    print(f"\n--- {rule.name} ---")
    show_predictions(rule)

for rule in TRIGRAM_RULES[:2]:
    print(f"\n--- {rule.name} ---")
    show_predictions(rule)

for rule in SKIP_BIGRAM_RULES[:2]:
    print(f"\n--- {rule.name} ---")
    show_predictions(rule)

for rule in INDUCTION_RULES[:1]:
    print(f"\n--- {rule.name} ---")
    show_predictions(rule)

# %%

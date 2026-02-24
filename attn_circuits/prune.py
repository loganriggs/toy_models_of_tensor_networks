"""
Iterative pruning & finetuning for BilinearGPT.

1. Load trained checkpoint
2. Fine-tune with L1 regularization
3. Remove lowest 1% of non-zero weights (permanently)
4. Continue training with L1 + masks
5. Repeat, saving checkpoints and per-rule losses at each step

Key invariants:
  - Once a weight is zeroed, it stays zero forever (binary mask)
  - When selecting "lowest 1%", only non-zero weights are candidates
  - Sanity checks verify zero count only increases

Usage:
    python attn_circuits/prune.py
    python attn_circuits/prune.py --model_tag depth1_1L_4H_d64_5000steps --prune_pct 0.01
"""

import argparse
import os
import sys
import json
import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import BilinearGPT, Config, Muon
from generator import (
    LanguageGenerator, DEPTH1_RULES, DEPTH1_MIXING,
)
from train import make_batch, per_rule_loss

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Pruning helpers
# ---------------------------------------------------------------------------

def get_prunable_params(model):
    """Return OrderedDict of (name, param) for prunable 2D weight matrices.
    Excludes embeddings and lm_head."""
    prunable = {}
    for name, p in model.named_parameters():
        if 'embed' in name or 'lm_head' in name:
            continue
        if p.ndim == 2:
            prunable[name] = p
    return prunable


def make_masks(prunable_params, device):
    """Create binary masks (all True initially) for each prunable parameter."""
    return {name: torch.ones_like(p, dtype=torch.bool, device=device)
            for name, p in prunable_params.items()}


def apply_masks(prunable_params, masks):
    """Zero out pruned weights (enforce mask). Call after every optimizer step."""
    with torch.no_grad():
        for name, p in prunable_params.items():
            p.mul_(masks[name].float())


def zero_pruned_grads(prunable_params, masks):
    """Zero gradients for pruned weights before optimizer step."""
    for name, p in prunable_params.items():
        if p.grad is not None:
            p.grad[~masks[name]] = 0.0


def prune_lowest_pct(prunable_params, masks, pct=0.01):
    """Remove the lowest pct (of TOTAL prunable weights) from remaining non-zero weights.

    Returns:
        n_pruned: number of new weights pruned this step
        total_zero: total zero count after pruning
        total_params: total prunable parameter count
    """
    total_params = sum(p.numel() for p in prunable_params.values())
    n_to_prune = max(1, int(total_params * pct))

    # Collect all magnitudes; set already-pruned to inf so they're never selected
    all_mags = []
    param_names = sorted(prunable_params.keys())
    for name in param_names:
        p = prunable_params[name]
        m = masks[name]
        mag = p.detach().abs().clone()
        mag[~m] = float('inf')
        all_mags.append(mag.flatten())

    all_mags = torch.cat(all_mags)
    n_remaining = (all_mags < float('inf')).sum().item()
    n_to_prune = min(n_to_prune, n_remaining)

    if n_to_prune == 0:
        total_zero = sum((~m).sum().item() for m in masks.values())
        return 0, total_zero, total_params

    # Find the n_to_prune smallest magnitudes
    _, indices = all_mags.topk(n_to_prune, largest=False)
    prune_flat = torch.zeros_like(all_mags, dtype=torch.bool)
    prune_flat[indices] = True

    # Map back to per-parameter masks
    offset = 0
    n_pruned = 0
    with torch.no_grad():
        for name in param_names:
            p = prunable_params[name]
            numel = p.numel()
            param_prune = prune_flat[offset:offset + numel].view(p.shape)
            offset += numel

            n_pruned += param_prune.sum().item()
            masks[name] = masks[name] & ~param_prune
            p[param_prune] = 0.0

    total_zero = sum((~m).sum().item() for m in masks.values())
    return int(n_pruned), int(total_zero), total_params


def count_zeros(masks):
    return sum((~m).sum().item() for m in masks.values())


def sparsity_pct(masks, prunable_params):
    total = sum(p.numel() for p in prunable_params.values())
    zeros = count_zeros(masks)
    return 100.0 * zeros / total if total > 0 else 0.0


def sanity_check(prunable_params, masks, prev_zero_count):
    """Verify: (1) zero count only increases, (2) no masked weight is non-zero."""
    current_zeros = count_zeros(masks)
    assert current_zeros >= prev_zero_count, \
        f"BUG: Zero count decreased! {prev_zero_count} -> {current_zeros}"

    # Check that all masked-out weights are actually zero
    for name, p in prunable_params.items():
        mask = masks[name]
        violations = (p[~mask] != 0).sum().item()
        assert violations == 0, \
            f"BUG: {violations} masked weights in {name} are non-zero!"

    return current_zeros


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, val_generator, device, seq_len=64, n_batches=20, batch_size=128):
    """Quick eval: overall loss + per-rule-class mean losses."""
    model.eval()
    all_losses = []
    rule_losses = defaultdict(list)

    with torch.no_grad():
        for _ in range(n_batches):
            x, y, labels = make_batch(val_generator, batch_size, seq_len, device)
            logits, loss = model(x, targets=y)
            all_losses.append(loss.item())
            rl = per_rule_loss(logits, y, labels)
            for k, v in rl.items():
                rule_losses[k].append(v)

    overall = np.mean(all_losses)
    per_rule = {k: np.mean(v) for k, v in rule_losses.items()}

    # Aggregate by rule class
    class_losses = {}
    for k, v in per_rule.items():
        if k.startswith('bigram'):
            cls = 'bigram'
        elif k.startswith('trigram'):
            cls = 'trigram'
        elif k.startswith('skip'):
            cls = 'skip_bigram'
        elif k in ('setup', 'noise'):
            cls = k
        else:
            cls = 'other'
        class_losses.setdefault(cls, []).append(v)

    class_means = {k: np.mean(v) for k, v in class_losses.items()}
    return overall, per_rule, class_means


# ---------------------------------------------------------------------------
# Optimizer (same split as train.py but with lower LR for finetuning)
# ---------------------------------------------------------------------------

def make_finetune_optimizer(model, lr):
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and 'embed' not in name and 'lm_head' not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=lr * 0.1, momentum=0.95))
    if adam_params:
        optimizers.append(torch.optim.AdamW(adam_params, lr=lr, weight_decay=0.0))
    return optimizers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Iterative pruning & finetuning')
    parser.add_argument('--model_tag', type=str, default='depth1_1L_4H_d64_5000steps')
    parser.add_argument('--prune_pct', type=float, default=0.01,
                        help='Fraction of TOTAL prunable weights to remove each iteration')
    parser.add_argument('--finetune_steps', type=int, default=200,
                        help='Training steps between each pruning round')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for finetuning (lower than initial training)')
    parser.add_argument('--l1_coeff', type=float, default=1e-4,
                        help='L1 regularization coefficient on prunable weights')
    parser.add_argument('--max_sparsity', type=float, default=0.95,
                        help='Stop when sparsity reaches this fraction')
    parser.add_argument('--kl_mode', action='store_true',
                        help='Use KL divergence with original model instead of CE loss')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Load model ---
    model_path = os.path.join(ROOT_DIR, 'models', f'bilinear_gpt_{args.model_tag}.pt')
    config_path = os.path.join(ROOT_DIR, 'configs', f'bilinear_gpt_{args.model_tag}.json')

    with open(config_path) as f:
        cd = json.load(f)
    config = Config(
        vocab_size=cd['vocab_size'], n_layer=cd['n_layer'], n_head=cd['n_head'],
        n_embd=cd['n_embd'], seq_len=cd['seq_len'],
        block_has_mlp=tuple(cd.get('block_has_mlp', [False] * cd['n_layer'])),
        use_rmsnorm=cd.get('use_rmsnorm', False),
        use_qk_norm=cd.get('use_qk_norm', False),
        use_final_norm=cd.get('use_final_norm', False),
    )
    model = BilinearGPT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded {args.model_tag}: {model.count_params():,} params")

    # --- Teacher model for KL mode ---
    teacher = None
    if args.kl_mode:
        teacher = BilinearGPT(config).to(device)
        teacher.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print("KL mode: using frozen original model as teacher")

    # --- Data generators ---
    generator = LanguageGenerator(rules=DEPTH1_RULES, mixing_weights=DEPTH1_MIXING,
                                  mode='mixed', seed=args.seed)
    val_generator = LanguageGenerator(rules=DEPTH1_RULES, mixing_weights=DEPTH1_MIXING,
                                      mode='mixed', seed=args.seed + 1)

    # --- Setup pruning ---
    prunable = get_prunable_params(model)
    masks = make_masks(prunable, device)
    total_prunable = sum(p.numel() for p in prunable.values())
    prev_zeros = 0

    mode_str = "KL divergence" if args.kl_mode else "CE loss"
    print(f"Prunable parameters: {total_prunable:,}")
    print(f"Prune {args.prune_pct*100:.1f}% ({int(total_prunable * args.prune_pct)}) per iteration")
    print(f"Finetune {args.finetune_steps} steps between prunes, lr={args.lr}, L1={args.l1_coeff}, loss={mode_str}")
    print()

    # --- Eval before pruning ---
    overall_loss, per_rule, class_means = evaluate(model, val_generator, device, args.seq_len)
    print(f"{'Iter':>4s} {'Sparsity':>8s} {'Loss':>7s} {'Bigram':>8s} {'Trigram':>8s} "
          f"{'Skip':>8s} {'Setup':>8s} {'Noise':>8s} {'Status'}")
    print("-" * 85)

    def status_str(cm):
        parts = []
        if cm.get('bigram', 0) > 0.5:
            parts.append('bigram DEGRADED')
        if cm.get('trigram', 0) > 0.5:
            parts.append('trigram DEGRADED')
        if cm.get('skip_bigram', 0) > 0.5:
            parts.append('skip DEGRADED')
        return '; '.join(parts) if parts else 'OK'

    def print_row(iteration, sp, ol, cm):
        print(f"{iteration:>4d} {sp:>7.1f}% {ol:>7.4f} {cm.get('bigram',0):>8.4f} "
              f"{cm.get('trigram',0):>8.4f} {cm.get('skip_bigram',0):>8.4f} "
              f"{cm.get('setup',0):>8.4f} {cm.get('noise',0):>8.4f} {status_str(cm)}")

    sp = sparsity_pct(masks, prunable)
    print_row(0, sp, overall_loss, class_means)

    # --- Save directory ---
    suffix = '_kl' if args.kl_mode else ''
    save_dir = os.path.join(ROOT_DIR, 'models', f'pruned{suffix}_{args.model_tag}')
    os.makedirs(save_dir, exist_ok=True)

    # Store results for summary
    results = [{
        'iteration': 0, 'sparsity': sp, 'loss': overall_loss,
        'class_means': class_means, 'per_rule': per_rule,
    }]

    # Save initial (unpruned) checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'masks': {k: v.cpu() for k, v in masks.items()},
        'iteration': 0, 'sparsity': sp,
    }, os.path.join(save_dir, 'iter_000_sp0.0.pt'))

    # --- Iterative prune & finetune ---
    max_iters = int(args.max_sparsity / args.prune_pct)
    optimizers = make_finetune_optimizer(model, args.lr)

    for iteration in range(1, max_iters + 1):
        # === Finetune with L1 ===
        model.train()
        for step in range(args.finetune_steps):
            x, y, labels = make_batch(generator, args.batch_size, args.seq_len, device)
            logits, ce_loss = model(x, targets=y)

            if args.kl_mode and teacher is not None:
                # KL(teacher || student): match student's distribution to teacher's
                with torch.no_grad():
                    teacher_logits, _ = teacher(x, targets=y)
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(logits, dim=-1)
                # KL(P || Q) = sum P * (log P - log Q)
                kl_loss = F.kl_div(student_log_probs, teacher_log_probs,
                                   log_target=True, reduction='batchmean')
                task_loss = kl_loss
            else:
                task_loss = ce_loss

            # L1 on prunable weights (only non-masked, but masked are 0 so contribute 0 anyway)
            l1_loss = sum(p.abs().sum() for p in prunable.values())
            total_loss = task_loss + args.l1_coeff * l1_loss

            for opt in optimizers:
                opt.zero_grad()
            total_loss.backward()

            # Zero gradients for pruned weights BEFORE optimizer step
            zero_pruned_grads(prunable, masks)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for opt in optimizers:
                opt.step()

            # Re-apply masks AFTER optimizer step (belt and suspenders)
            apply_masks(prunable, masks)

        # === Sanity check after finetuning ===
        prev_zeros = sanity_check(prunable, masks, prev_zeros)

        # === Prune ===
        n_pruned, total_zero, total_params = prune_lowest_pct(prunable, masks, args.prune_pct)

        # === Sanity check after pruning ===
        prev_zeros = sanity_check(prunable, masks, prev_zeros)

        sp = sparsity_pct(masks, prunable)

        # === Evaluate ===
        overall_loss, per_rule, class_means = evaluate(model, val_generator, device, args.seq_len)
        print_row(iteration, sp, overall_loss, class_means)

        results.append({
            'iteration': iteration, 'sparsity': sp, 'loss': overall_loss,
            'class_means': dict(class_means), 'per_rule': dict(per_rule),
            'n_pruned': n_pruned,
        })

        # === Save checkpoint ===
        ckpt_name = f'iter_{iteration:03d}_sp{sp:.1f}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'masks': {k: v.cpu() for k, v in masks.items()},
            'iteration': iteration, 'sparsity': sp,
            'class_means': dict(class_means),
            'per_rule': dict(per_rule),
            'overall_loss': overall_loss,
        }, os.path.join(save_dir, ckpt_name))

        # Check if we should stop
        if sp >= args.max_sparsity * 100:
            print(f"\nReached {sp:.1f}% sparsity, stopping.")
            break

    # --- Save full results ---
    results_path = os.path.join(save_dir, 'pruning_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")
    print(f"Checkpoints saved in: {save_dir}")

    # --- Print summary table ---
    print("\n" + "=" * 85)
    print("PRUNING SUMMARY")
    print("=" * 85)
    print(f"{'Iter':>4s} {'Sparsity':>8s} {'Loss':>7s} {'Bigram':>8s} {'Trigram':>8s} "
          f"{'Skip':>8s} {'Status'}")
    print("-" * 60)
    for r in results:
        cm = r['class_means']
        print(f"{r['iteration']:>4d} {r['sparsity']:>7.1f}% {r['loss']:>7.4f} "
              f"{cm.get('bigram',0):>8.4f} {cm.get('trigram',0):>8.4f} "
              f"{cm.get('skip_bigram',0):>8.4f} {status_str(cm)}")


if __name__ == '__main__':
    main()

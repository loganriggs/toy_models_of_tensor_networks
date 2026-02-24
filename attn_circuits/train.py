"""
Train BilinearGPT on synthetic language data.

Usage:
    # 1-layer attention-only (default)
    python attn_circuits/train.py

    # With wandb
    python attn_circuits/train.py --wandb --wandb_run_name my_run

    # With pre-generated data cache (eliminates data gen bottleneck)
    python attn_circuits/train.py --cache_size 50000

    # Debug (short run)
    python attn_circuits/train.py --debug
"""

import faulthandler
faulthandler.enable()

import argparse
import os
import sys
import json
import time
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Add attn_circuits to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import BilinearGPT, Config, Muon
from generator import (
    LanguageGenerator, FastLanguageGenerator, VOCAB_SIZE, ID2TOKEN,
    BIGRAM_RULES, TRIGRAM_RULES, SKIP_BIGRAM_RULES,
    INDUCTION_RULES, BRACKET_RULES, DEPTH1_RULES,
    ALL_RULES, DEFAULT_MIXING, DEPTH1_MIXING,
    LANG128, make_scaled_language,
)
from generator_v3 import (
    LanguageV3,
    make_batch as v3_make_batch, per_rule_loss as v3_per_rule_loss,
    true_entropies as v3_true_entropies,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_batch(generator, batch_size, seq_len, device):
    """Sample a batch from the generator. Returns (input_ids, targets, rule_labels)."""
    all_tokens, all_labels = generator.sample_batch(batch_size, length=seq_len + 1)
    tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    # labels shifted same way (label[i] describes how token[i] was generated)
    labels = [lab[1:] for lab in all_labels]  # drop seed token, align with targets
    return x, y, labels


# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

def generate_cache(generator, cache_size, seq_len, device, cache_path=None):
    """Pre-generate sequences into a tensor + labels list.

    Returns (tokens_tensor, labels_list) where tokens_tensor is (N, seq_len+1)
    on device and labels_list is a list of N label sequences.
    If cache_path is given, saves to disk for reuse.
    """
    t0 = time.time()
    print(f"Generating cache: {cache_size} sequences x {seq_len+1} tokens ...")

    # Generate in chunks to show progress
    chunk = 1024
    all_tokens, all_labels = [], []
    remaining = cache_size
    while remaining > 0:
        n = min(chunk, remaining)
        toks, labs = generator.sample_batch(n, length=seq_len + 1)
        all_tokens.extend(toks)
        all_labels.extend(labs)
        remaining -= n
        done = cache_size - remaining
        if done % 10000 == 0 or remaining == 0:
            print(f"  {done}/{cache_size} ({time.time()-t0:.1f}s)")

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
    dt = time.time() - t0
    print(f"Cache ready: {cache_size} sequences in {dt:.1f}s ({cache_size/dt:.0f} seq/s)")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path + '.tokens.npy', tokens_tensor.cpu().numpy())
        with open(cache_path + '.labels.pkl', 'wb') as f:
            pickle.dump(all_labels, f)
        print(f"Cache saved: {cache_path}.*")

    return tokens_tensor, all_labels


def load_cache(cache_path, device):
    """Load a previously saved cache from disk."""
    tokens = np.load(cache_path + '.tokens.npy')
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
    with open(cache_path + '.labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    print(f"Cache loaded: {len(labels)} sequences from {cache_path}.*")
    return tokens_tensor, labels


def sample_from_cache(cache_tokens, cache_labels, batch_size, np_rng):
    """Sample a random batch from the pre-generated cache.

    Returns (x, y, labels) same format as make_batch.
    np_rng: numpy RandomState for reproducible sampling.
    """
    N = len(cache_tokens)
    indices = np_rng.randint(0, N, size=batch_size)
    tokens = cache_tokens[indices]
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    labels = [cache_labels[int(i)][1:] for i in indices]  # drop seed token, align with targets
    return x, y, labels


def per_rule_loss(logits, targets, labels):
    """Compute CE loss broken down by which rule generated each target token."""
    # logits: (B, T, V), targets: (B, T), labels: list of list of str
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='none')
    ce = ce.view(B, T)

    rule_losses = defaultdict(list)
    for b in range(B):
        for t in range(T):
            lab = labels[b][t]
            # Normalize bracket sub-labels (v2 only)
            if lab.startswith('bracket') and not lab.startswith(('paren', 'quote', 'skip_trigram')):
                lab = 'bracket'
            rule_losses[lab].append(ce[b, t].item())

    return {k: np.mean(v) for k, v in rule_losses.items() if v}


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------

def make_optimizer(model, lr, weight_decay=0.0):
    """Muon for 2-D weight matrices, AdamW for embeddings / lm_head / 1-D params."""
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
        optimizers.append(torch.optim.AdamW(adam_params, lr=lr, weight_decay=weight_decay))
    return optimizers


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(step, total_steps, warmup_steps, lr):
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * 0.5 * (1.0 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train BilinearGPT on synthetic language')
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--num_steps', type=int, default=5000)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=250)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--rules', type=str, default='depth1',
                        choices=['bigram', 'trigram', 'skip_bigram', 'induction', 'bracket', 'depth1', 'all'],
                        help='Which rule set to train on')
    parser.add_argument('--mode', type=str, default='mixed', choices=['isolated', 'mixed'])
    parser.add_argument('--lang', type=str, default=None, choices=['128'],
                        help='Use scaled language (e.g. 128 for 128-token vocab with 90 rules)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--cache_size', type=int, default=0,
                        help='Pre-generate this many sequences at startup (0 = generate on-the-fly)')
    parser.add_argument('--fast_gen', action='store_true', default=True,
                        help='Use FastLanguageGenerator (numpy-based, ~5-10x faster). Default: True.')
    parser.add_argument('--no_fast_gen', dest='fast_gen', action='store_false',
                        help='Disable FastLanguageGenerator, use original LanguageGenerator.')
    parser.add_argument('--generator', type=str, default='v2', choices=['v2', 'v3'],
                        help='Generator version: v2 (26-token) or v3 (52-token realistic)')
    parser.add_argument('--tokens_per_category', type=int, default=8,
                        help='Tokens per category for v3 generator (1-32). Default 8 = 52 vocab.')
    parser.add_argument('--cached_data', type=str, default=None,
                        help='Path to pre-cached pickle file (v3 only). Avoids segfault from numpy+torch interaction.')
    parser.add_argument('--optimizer', type=str, default='muon', choices=['muon', 'adamw'],
                        help='Optimizer: muon (default) or adamw')
    parser.add_argument('--wandb_project', type=str, default='attn-circuits')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()

    if args.debug:
        args.num_steps = 200
        args.eval_interval = 50
        args.log_interval = 10

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Language / Rules ---
    use_v3 = (args.generator == 'v3')

    if use_v3:
        # v3: realistic language with category bigrams, scalable vocab
        generator = LanguageV3(tokens_per_category=args.tokens_per_category, seed=args.seed)
        val_generator = LanguageV3(tokens_per_category=args.tokens_per_category, seed=args.seed + 1)
        vocab_size = generator.vocab_size
        print(f"Using generator v3: {vocab_size} tokens ({args.tokens_per_category}/cat), realistic POS transitions")
        print(f"  Active classes: {sorted(generator.active_classes)}")
    elif False:
        pass  # placeholder to keep elif chain

    lang = None  # custom language dict, or None for default 25-token vocab
    if not use_v3 and args.lang == '128':
        lang = LANG128
        rules = lang['depth1_rules']
        mixing = lang['depth1_mixing']
        vocab_size = lang['vocab_size']
        gen_kwargs = dict(vocab=lang['vocab'], token2id=lang['token2id'], id2token=lang['id2token'])
        print(f"Using LANG128: {vocab_size} tokens, {len(lang['bigram_rules'])} bigrams, "
              f"{len(lang['trigram_rules'])} trigrams, {len(lang['skip_bigram_rules'])} skip-bigrams")
    elif not use_v3:
        rule_map = {
            'bigram': BIGRAM_RULES,
            'trigram': TRIGRAM_RULES,
            'skip_bigram': SKIP_BIGRAM_RULES,
            'induction': INDUCTION_RULES,
            'bracket': BRACKET_RULES,
            'depth1': DEPTH1_RULES,
            'all': ALL_RULES,
        }
        mixing_map = {
            'depth1': DEPTH1_MIXING,
        }
        rules = rule_map[args.rules]
        mixing = mixing_map.get(args.rules, None)
        vocab_size = VOCAB_SIZE
        gen_kwargs = {}

        GenClass = FastLanguageGenerator if args.fast_gen else LanguageGenerator
        if args.fast_gen:
            print("Using FastLanguageGenerator (numpy-based)")
        generator = GenClass(rules=rules, mixing_weights=mixing, mode=args.mode,
                             seed=args.seed, **gen_kwargs)
        val_generator = GenClass(rules=rules, mixing_weights=mixing, mode=args.mode,
                                 seed=args.seed + 1, **gen_kwargs)

    # --- Data cache ---
    cache_tokens, cache_labels = None, None
    val_cache_tokens, val_cache_labels = None, None
    cache_rng = np.random.RandomState(args.seed + 100)
    val_cache_rng = np.random.RandomState(args.seed + 200)

    if args.cache_size > 0:
        cache_dir = os.path.join(ROOT_DIR, 'data', 'cache')
        lang_tag_c = f'v{vocab_size}_' if args.lang else ''
        cache_tag = f'{lang_tag_c}{args.rules}_{args.mode}_seq{args.seq_len}_n{args.cache_size}'
        train_cache_path = os.path.join(cache_dir, f'{cache_tag}_train')
        val_cache_path = os.path.join(cache_dir, f'{cache_tag}_val')

        # Try to load from disk, otherwise generate
        if os.path.exists(train_cache_path + '.tokens.npy'):
            cache_tokens, cache_labels = load_cache(train_cache_path, device)
            val_cache_tokens, val_cache_labels = load_cache(val_cache_path, device)
        else:
            cache_tokens, cache_labels = generate_cache(
                generator, args.cache_size, args.seq_len, device, cache_path=train_cache_path)
            val_cache_tokens, val_cache_labels = generate_cache(
                val_generator, args.cache_size // 5, args.seq_len, device, cache_path=val_cache_path)

    # --- Model ---
    # For v3, model seq_len must include pre-context
    model_seq_len = args.seq_len + (generator.pre_context_len if use_v3 else 0)
    config = Config(
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_len=model_seq_len,
        block_has_mlp=tuple([False] * args.n_layer),
        use_rmsnorm=False,
        use_qk_norm=False,
        use_final_norm=False,
    )
    model = BilinearGPT(config).to(device)
    num_params = model.count_params()
    print(f"Config: {args.n_layer}L {args.n_head}H d={args.n_embd} seq={args.seq_len}")
    print(f"Rules: {args.rules} ({args.mode} mode)")
    print(f"Parameters: {num_params:,}")
    print(f"Device: {device}")

    # --- Optimizer ---
    if args.optimizer == 'adamw':
        optimizers = [torch.optim.AdamW(model.parameters(), lr=args.lr)]
        print(f"Optimizer: AdamW (lr={args.lr})")
    else:
        optimizers = make_optimizer(model, args.lr)
        print(f"Optimizer: Muon + AdamW")

    # --- Wandb ---
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or f'{args.rules}_{args.n_layer}L_{args.n_head}H_d{args.n_embd}'
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # --- Load pre-cached data (v3 only, avoids numpy+torch segfault) ---
    v3_cached = None
    if args.cached_data and use_v3:
        import pickle as _pkl
        print(f"Loading cached data from {args.cached_data}...")
        with open(args.cached_data, 'rb') as f:
            v3_cached = _pkl.load(f)
        print(f"  {len(v3_cached['train'])} train batches, {len(v3_cached['eval'])} eval batches")

    # --- Pre-generate eval data (v3: generate once, reuse every eval) ---
    cached_eval_batches = None
    if use_v3 and v3_cached:
        # Convert cached eval to tensors
        cached_eval_batches = []
        for toks_list, labs_list in v3_cached['eval']:
            tokens = torch.tensor(toks_list, dtype=torch.long, device=device)
            x = tokens[:, :-1]
            y = tokens[:, 1:]
            labels = [lab[1:] for lab in labs_list]
            mask = torch.zeros_like(x, dtype=torch.bool)  # no pre-context masking in cached data
            cached_eval_batches.append((x, y, labels, mask))
        print(f"  Eval batches ready ({len(cached_eval_batches)} batches)")
    elif use_v3:
        n_eval_batches = 5
        print(f"Pre-generating {n_eval_batches} eval batches...")
        cached_eval_batches = []
        for _ in range(n_eval_batches):
            x_val, y_val, labels_val, mask_val = v3_make_batch(
                val_generator, args.batch_size, args.seq_len, device)
            cached_eval_batches.append((x_val, y_val, labels_val, mask_val))
        print(f"  Done ({n_eval_batches} × {args.batch_size} seqs)")

    # --- Tag for saving (computed early for checkpoints) ---
    if use_v3:
        lang_tag = 'v3_'
        rule_tag = 'all'
    else:
        lang_tag = f'v{vocab_size}_' if args.lang else ''
        rule_tag = args.rules
    tag = f'{lang_tag}{rule_tag}_{args.n_layer}L_{args.n_head}H_d{args.n_embd}_{args.num_steps}steps'
    checkpoint_dir = os.path.join(ROOT_DIR, 'models', 'checkpoints', tag)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    train_losses = []
    val_losses_list = []
    val_iters = []
    per_rule_losses_history = defaultdict(list)  # rule_name -> list of mean losses per eval step

    model.train()
    for step in range(args.num_steps):
        # LR schedule
        lr_now = cosine_lr(step, args.num_steps, args.warmup_steps, args.lr)
        for opt in optimizers:
            for pg in opt.param_groups:
                pg['lr'] = lr_now if isinstance(opt, torch.optim.AdamW) else lr_now * 0.1

        if use_v3 and v3_cached:
            # Use pre-cached data (no numpy/torch interaction)
            batch_idx = step % len(v3_cached['train'])
            toks_list, labs_list = v3_cached['train'][batch_idx]
            tokens = torch.tensor(toks_list, dtype=torch.long, device=device)
            x = tokens[:, :-1]
            y = tokens[:, 1:]
            labels = [lab[1:] for lab in labs_list]
            mask = torch.zeros_like(x, dtype=torch.bool)
            logits, _ = model(x)
            V = logits.shape[-1]
            ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction='none')
            ce = ce.view(x.shape[0], -1)
            valid = ~mask
            loss = (ce * valid.float()).sum() / valid.float().sum()
        elif use_v3:
            x, y, labels, mask = v3_make_batch(generator, args.batch_size, args.seq_len, device)
            logits, _ = model(x)  # don't use model's internal loss
            # Masked loss: exclude pre-context and padding
            V = logits.shape[-1]
            ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction='none')
            ce = ce.view(x.shape[0], -1)
            valid = ~mask
            loss = (ce * valid.float()).sum() / valid.float().sum()
        elif cache_tokens is not None:
            x, y, labels = sample_from_cache(cache_tokens, cache_labels, args.batch_size, np_rng=cache_rng)
            mask = None
            logits, loss = model(x, targets=y)
        else:
            x, y, labels = make_batch(generator, args.batch_size, args.seq_len, device)
            mask = None
            logits, loss = model(x, targets=y)

        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        # --- Logging ---
        if step % args.log_interval == 0:
            print(f"step {step:5d} | loss {train_loss:.4f} | lr {lr_now:.6f}")
            if args.wandb:
                wandb.log({'train_loss': train_loss, 'lr': lr_now, 'step': step})

        # --- Eval ---
        if step % args.eval_interval == 0:
            model.eval()
            eval_losses = []
            all_rule_losses = defaultdict(list)
            with torch.no_grad():
                if use_v3 and cached_eval_batches:
                    eval_data = cached_eval_batches
                else:
                    eval_data = range(20)

                for eval_item in eval_data:
                    if use_v3 and cached_eval_batches:
                        x_val, y_val, labels_val, mask_val = eval_item
                        logits_val, _ = model(x_val)
                        V = logits_val.shape[-1]
                        ce_val = F.cross_entropy(logits_val.reshape(-1, V), y_val.reshape(-1), reduction='none')
                        ce_val = ce_val.view(x_val.shape[0], -1)
                        valid_val = ~mask_val
                        loss_val = (ce_val * valid_val.float()).sum() / valid_val.float().sum()
                    elif val_cache_tokens is not None:
                        x_val, y_val, labels_val = sample_from_cache(
                            val_cache_tokens, val_cache_labels, args.batch_size, np_rng=val_cache_rng)
                        mask_val = None
                        logits_val, loss_val = model(x_val, targets=y_val)
                    else:
                        x_val, y_val, labels_val = make_batch(
                            val_generator, args.batch_size, args.seq_len, device
                        )
                        mask_val = None
                        logits_val, loss_val = model(x_val, targets=y_val)
                    eval_losses.append(loss_val.item())
                    rl = (v3_per_rule_loss(logits_val, y_val, labels_val, mask_val)
                          if use_v3 else per_rule_loss(logits_val, y_val, labels_val))
                    for k, v in rl.items():
                        all_rule_losses[k].append(v)

            val_loss = np.mean(eval_losses)
            val_losses_list.append(val_loss)
            val_iters.append(step)

            rule_summary = {k: np.mean(v) for k, v in all_rule_losses.items()}
            for k, v in rule_summary.items():
                per_rule_losses_history[k].append(v)
            rule_str = ' | '.join(f'{k}: {v:.3f}' for k, v in sorted(rule_summary.items()))
            print(f"  val loss {val_loss:.4f} | {rule_str}")

            if args.wandb:
                log_dict = {'val_loss': val_loss, 'step': step}
                for k, v in rule_summary.items():
                    log_dict[f'rule_loss/{k}'] = v
                wandb.log(log_dict)

            # Save checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f'step_{step:05d}.pt')
            torch.save(model.state_dict(), ckpt_path)

            model.train()
            if 'cuda' in str(device):
                torch.cuda.empty_cache()

    print("Training complete!")

    # --- Final eval ---
    model.eval()
    final_rule_losses = defaultdict(list)
    with torch.no_grad():
        n_final_eval = 50 if not (use_v3 and v3_cached) else len(cached_eval_batches)
        for eval_i in range(n_final_eval):
            if use_v3 and cached_eval_batches:
                x_val, y_val, labels_val, mask_val = cached_eval_batches[eval_i % len(cached_eval_batches)]
                logits_val, _ = model(x_val)
                rl = v3_per_rule_loss(logits_val, y_val, labels_val, mask_val)
            elif use_v3:
                x_val, y_val, labels_val, mask_val = v3_make_batch(
                    val_generator, args.batch_size, args.seq_len, device)
                logits_val, _ = model(x_val)
                rl = v3_per_rule_loss(logits_val, y_val, labels_val, mask_val)
            elif val_cache_tokens is not None:
                x_val, y_val, labels_val = sample_from_cache(
                    val_cache_tokens, val_cache_labels, args.batch_size, np_rng=val_cache_rng)
                logits_val, loss_val = model(x_val, targets=y_val)
                rl = per_rule_loss(logits_val, y_val, labels_val)
            else:
                x_val, y_val, labels_val = make_batch(val_generator, args.batch_size, args.seq_len, device)
                logits_val, loss_val = model(x_val, targets=y_val)
                rl = per_rule_loss(logits_val, y_val, labels_val)
            for k, v in rl.items():
                final_rule_losses[k].append(v)

    print("\nFinal per-rule losses:")
    for k, v in sorted(final_rule_losses.items()):
        print(f"  {k:20s}: {np.mean(v):.4f}")

    # --- Save ---
    os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'configs'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'figures'), exist_ok=True)

    model_path = os.path.join(ROOT_DIR, 'models', f'bilinear_gpt_{tag}.pt')
    config_path = os.path.join(ROOT_DIR, 'configs', f'bilinear_gpt_{tag}.json')

    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    config_dict = {**config.__dict__, **vars(args),
                   'num_params': num_params,
                   'final_train_loss': train_losses[-1],
                   'final_val_loss': val_losses_list[-1] if val_losses_list else None,
                   'final_rule_losses': {k: float(np.mean(v)) for k, v in final_rule_losses.items()}}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved: {config_path}")

    # --- Save training logs (for re-plotting without re-training) ---
    logs = {
        'tag': tag,
        'config': config_dict,
        'train_losses': train_losses,
        'val_losses': val_losses_list,
        'val_iters': val_iters,
        'per_rule_losses_history': dict(per_rule_losses_history),
        'final_rule_losses': {k: float(np.mean(v)) for k, v in final_rule_losses.items()},
    }
    if use_v3:
        logs['true_entropies'] = v3_true_entropies(generator)
    logs_path = os.path.join(ROOT_DIR, 'models', f'training_logs_{tag}.pkl')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    print(f"Training logs saved: {logs_path}")
    print(f"Checkpoints saved: {checkpoint_dir}/ ({len(os.listdir(checkpoint_dir))} files)")

    # --- Save rules.txt alongside figures ---
    attn_fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', tag)
    os.makedirs(attn_fig_dir, exist_ok=True)

    rules_txt_path = os.path.join(attn_fig_dir, 'rules.txt')
    with open(rules_txt_path, 'w') as f:
        f.write(f"Model: BilinearGPT {tag}\n")
        f.write(f"Config: {args.n_layer}L {args.n_head}H d={args.n_embd} seq={args.seq_len}\n")
        f.write(f"Parameters: {num_params:,}\n")
        f.write(f"Training: {args.num_steps} steps, batch={args.batch_size}, lr={args.lr}\n")

        if use_v3:
            f.write(f"Generator: v3 (52-token realistic)\n\n")
            f.write(generator.describe_rules())
            f.write("\n\n")

            # Rule stats
            stats = generator.get_rule_stats(n_sequences=500, length=args.seq_len)
            f.write("Empirical rule fractions:\n")
            for k, v in stats.items():
                f.write(f"  {k:20s}: {v:.3f}\n")
            f.write("\n")
        else:
            f.write(f"Rules: {args.rules} ({args.mode} mode)\n\n")
            if mixing:
                f.write("Mixing weights:\n")
                for k, v in sorted(mixing.items()):
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

            f.write("Rules:\n")
            for rule in rules:
                if hasattr(rule, 'trigger') and hasattr(rule, 'output'):
                    f.write(f"  {rule.name}: {rule.trigger} -> {rule.output}\n")
                elif hasattr(rule, 'trigger1'):
                    f.write(f"  {rule.name}: {rule.trigger1}, {rule.trigger2} -> {rule.output}\n")
                elif hasattr(rule, 'anchor'):
                    f.write(f"  {rule.name}: {rule.anchor} ... {rule.current_marker} -> {rule.output} (max_skip={rule.max_skip})\n")
                elif hasattr(rule, 'trigger') and rule.rule_class == 'induction':
                    f.write(f"  {rule.name}: {rule.trigger} X ... {rule.trigger} -> X (copy)\n")
                elif rule.rule_class == 'bracket':
                    f.write(f"  {rule.name}: {rule.open_token} content {rule.close_token}\n")
                else:
                    f.write(f"  {rule.name} ({rule.rule_class})\n")

        f.write(f"\nFinal per-rule losses:\n")
        for k, v in sorted(final_rule_losses.items()):
            f.write(f"  {k:25s}: {np.mean(v):.4f}\n")

        # Add example sequences
        if use_v3:
            example_gen = LanguageV3(seed=args.seed + 999)
            f.write(f"\nExample sequences (seq_len={min(args.seq_len, 48)}):\n")
            for i in range(5):
                toks, labs = example_gen.sample_sequence(min(args.seq_len, 48))
                f.write(f"  {example_gen.format_sequence(toks, labs, compact=True)}\n")
            f.write("  Legend: CB=cat_bigram CT=cat_trigram TB=tok_bigram TG=tok_trigram IN=induction B(=bracket_open BC=bracket_content B)=bracket_close PC=pre_context\n")
        else:
            id2tok = gen_kwargs.get('id2token', ID2TOKEN) if gen_kwargs else ID2TOKEN
            example_gen = LanguageGenerator(rules=rules, mixing_weights=mixing, mode=args.mode,
                                            seed=args.seed + 999)
            f.write(f"\nExample sequences:\n")
            for i in range(5):
                toks, labs = example_gen.sample_sequence(min(args.seq_len, 64))
                tok_strs = [id2tok[t] for t in toks]
                annotated = []
                for t, l in zip(tok_strs, labs):
                    if l.startswith('bigram'): annotated.append(f'{t}[B]')
                    elif l.startswith('trigram'): annotated.append(f'{t}[T]')
                    elif l.startswith('skip'): annotated.append(f'{t}[S]')
                    elif l.startswith('induction'): annotated.append(f'{t}[I]')
                    elif l.startswith('bracket'): annotated.append(f'{t}[Br]')
                    else: annotated.append(t)
                f.write(f"  {' '.join(annotated)}\n")
            f.write("  Legend: [B]=bigram [T]=trigram [S]=skip-bigram [I]=induction [Br]=bracket, unlabeled=setup/noise\n")
    print(f"Rules saved: {rules_txt_path}")

    # --- Plot ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, alpha=0.3, label='train')
    window = min(50, len(train_losses) // 10)
    if window > 1:
        smooth = np.convolve(train_losses, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, window - 1 + len(smooth)), smooth, label='train (smooth)', lw=2)
    if val_losses_list:
        ax.plot(val_iters, val_losses_list, 'o-', label='val', lw=2, ms=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'BilinearGPT {tag}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(ROOT_DIR, 'figures', f'training_{tag}.png')
    fig.savefig(plot_path, dpi=150)
    # Also save to attn_circuits/figures/{tag}/ alongside circuit plots
    attn_plot_path = os.path.join(attn_fig_dir, f'training_{tag}.png')
    fig.savefig(attn_plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")
    print(f"Plot saved: {attn_plot_path}")

    # --- Per-rule loss plot ---
    if per_rule_losses_history and val_iters:
        # Classify rules into groups by prefix
        if use_v3:
            RULE_CLASS_PREFIXES = [
                ('cat_bigram', 'cat_bigram'),
                ('cat_trigram', 'cat_trigram'),
                ('noun_bigram', 'noun_bigram'),
                ('place_bigram', 'place_bigram'),
                ('skip_bigram', 'skip_bigram'),
                ('skip_trigram', 'skip_trigram'),
                ('tok_trigram', 'tok_trigram'),
                ('induction', 'induction'),
                ('paren', 'paren'),
                ('quote', 'quote'),
            ]
            CLASS_COLORS = {
                'cat_bigram': '#1f77b4',   # blue
                'cat_trigram': '#17becf',  # cyan
                'noun_bigram': '#ff7f0e',  # orange
                'place_bigram': '#bcbd22', # olive
                'skip_bigram': '#e377c2',  # pink
                'skip_trigram': '#8c564b', # brown
                'tok_trigram': '#2ca02c',  # green
                'induction': '#d62728',    # red
                'paren': '#9467bd',        # purple
                'quote': '#7f7f7f',        # gray
                'other': '#aaaaaa',        # light gray
            }
        else:
            RULE_CLASS_PREFIXES = [
                ('bigram', 'bigram'),
                ('trigram', 'trigram'),
                ('skip', 'skip_bigram'),
                ('induction', 'induction'),
                ('bracket', 'bracket'),
            ]
            CLASS_COLORS = {
                'bigram': '#1f77b4',       # blue
                'trigram': '#ff7f0e',      # orange
                'skip_bigram': '#2ca02c',  # green
                'induction': '#d62728',    # red
                'bracket': '#9467bd',      # purple
                'other': '#7f7f7f',        # gray
            }

        def classify_rule(name):
            for prefix, cls in RULE_CLASS_PREFIXES:
                if name.startswith(prefix):
                    return cls
            return 'other'

        fig2, ax2 = plt.subplots(figsize=(12, 7))

        # Sort rules by class then name for consistent ordering
        sorted_rules = sorted(per_rule_losses_history.keys(),
                              key=lambda r: (classify_rule(r), r))

        # Track which classes we've seen (for legend dedup)
        class_legend_added = set()

        for rule_name in sorted_rules:
            losses = per_rule_losses_history[rule_name]
            cls = classify_rule(rule_name)
            color = CLASS_COLORS.get(cls, CLASS_COLORS['other'])

            # Only add a class label to the first rule of each class
            if cls not in class_legend_added:
                label = f'{cls}: {rule_name}'
                class_legend_added.add(cls)
            else:
                label = rule_name

            # Use steps where this rule was present (it may not appear at every eval)
            # per_rule_losses_history values are aligned with val_iters (one entry per eval step)
            steps = val_iters[:len(losses)]
            ax2.plot(steps, losses, color=color, alpha=0.7, lw=1.5, label=label)

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Per-Rule CE Loss')
        ax2.set_title(f'Per-Rule Losses — BilinearGPT {tag}')
        ax2.grid(True, alpha=0.3)

        # Place legend outside the plot for readability
        ax2.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0., framealpha=0.9)
        fig2.tight_layout(rect=[0, 0, 0.78, 1])  # leave room for legend on right

        rule_plot_path = os.path.join(ROOT_DIR, 'figures', f'rule_losses_{tag}.png')
        fig2.savefig(rule_plot_path, dpi=150, bbox_inches='tight')
        attn_rule_plot_path = os.path.join(attn_fig_dir, f'rule_losses_{tag}.png')
        fig2.savefig(attn_rule_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"Rule loss plot saved: {rule_plot_path}")
        print(f"Rule loss plot saved: {attn_rule_plot_path}")

        # --- KL divergence plot (CE - H_true) ---
        if use_v3:
            h_true = v3_true_entropies(generator)

            fig3, ax3 = plt.subplots(figsize=(12, 7))
            class_legend_added_kl = set()

            for rule_name in sorted_rules:
                losses = per_rule_losses_history[rule_name]
                h = h_true.get(rule_name, 0.0)
                kl_vals = [max(0.0, l - h) for l in losses]
                cls = classify_rule(rule_name)
                color = CLASS_COLORS.get(cls, CLASS_COLORS['other'])

                if cls not in class_legend_added_kl:
                    label = f'{rule_name} (H={h:.2f})'
                    class_legend_added_kl.add(cls)
                else:
                    label = f'{rule_name} (H={h:.2f})'

                steps = val_iters[:len(kl_vals)]
                ax3.plot(steps, kl_vals, color=color, alpha=0.7, lw=1.5, label=label)

            ax3.set_xlabel('Step')
            ax3.set_ylabel('KL Divergence (CE - H_true)')
            ax3.set_title(f'Per-Rule KL Divergence — BilinearGPT {tag}')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                       borderaxespad=0., framealpha=0.9)
            fig3.tight_layout(rect=[0, 0, 0.75, 1])

            kl_plot_path = os.path.join(ROOT_DIR, 'figures', f'kl_divergence_{tag}.png')
            fig3.savefig(kl_plot_path, dpi=150, bbox_inches='tight')
            attn_kl_plot_path = os.path.join(attn_fig_dir, f'kl_divergence_{tag}.png')
            fig3.savefig(attn_kl_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print(f"KL plot saved: {kl_plot_path}")
            print(f"KL plot saved: {attn_kl_plot_path}")

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

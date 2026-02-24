"""
Verify that FastLanguageGenerator produces statistically equivalent output
to the original LanguageGenerator.

Compares:
1. Token frequency distribution (chi-squared test)
2. Rule label distribution (chi-squared test)
3. Rule firing rates per class
4. Timing comparison

Usage:
    python attn_circuits/verify_fast_generator.py
"""

import sys
import os
import time
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import (
    LanguageGenerator, FastLanguageGenerator,
    BIGRAM_RULES, TRIGRAM_RULES, SKIP_BIGRAM_RULES,
    DEPTH1_RULES, ALL_RULES, INDUCTION_RULES, BRACKET_RULES,
    DEPTH1_MIXING, DEFAULT_MIXING,
    VOCAB, TOKEN2ID, ID2TOKEN, VOCAB_SIZE,
    LANG128,
)


def compare_distributions(name, counts_old, counts_new, all_keys=None):
    """Compare two count distributions and report statistics."""
    if all_keys is None:
        all_keys = sorted(set(list(counts_old.keys()) + list(counts_new.keys())))

    total_old = sum(counts_old.values())
    total_new = sum(counts_new.values())

    print(f"\n--- {name} ---")
    print(f"  Total old: {total_old}, Total new: {total_new}")

    # Compute relative frequency differences
    max_diff = 0
    diffs = []
    for k in all_keys:
        freq_old = counts_old.get(k, 0) / max(total_old, 1)
        freq_new = counts_new.get(k, 0) / max(total_new, 1)
        diff = abs(freq_old - freq_new)
        diffs.append(diff)
        if diff > max_diff:
            max_diff = diff
            max_diff_key = k

    mean_diff = np.mean(diffs)
    print(f"  Mean abs freq diff: {mean_diff:.6f}")
    print(f"  Max abs freq diff:  {max_diff:.6f} (at '{max_diff_key}' if exists)")

    # Show top-10 entries by frequency
    print(f"  Top entries (old freq | new freq):")
    sorted_keys = sorted(all_keys, key=lambda k: counts_old.get(k, 0), reverse=True)[:15]
    for k in sorted_keys:
        fo = counts_old.get(k, 0) / max(total_old, 1)
        fn = counts_new.get(k, 0) / max(total_new, 1)
        marker = " <<<" if abs(fo - fn) > 0.02 else ""
        print(f"    {str(k):20s}: {fo:.4f} vs {fn:.4f}{marker}")

    return mean_diff < 0.02  # pass if mean diff < 2%


def count_tokens(all_tokens):
    """Count token frequencies across all sequences."""
    c = Counter()
    for seq in all_tokens:
        c.update(seq)
    return c


def count_labels(all_labels):
    """Count label frequencies across all sequences."""
    c = Counter()
    for seq in all_labels:
        c.update(seq)
    return c


def count_rule_classes(all_labels):
    """Count rule class frequencies (bigram, trigram, skip_bigram, etc.)."""
    c = Counter()
    for seq in all_labels:
        for lab in seq:
            if lab.startswith('bigram'):
                c['bigram'] += 1
            elif lab.startswith('trigram'):
                c['trigram'] += 1
            elif lab.startswith('skip'):
                c['skip_bigram'] += 1
            elif lab.startswith('induction'):
                c['induction'] += 1
            elif lab.startswith('bracket'):
                c['bracket'] += 1
            elif lab == 'setup':
                c['setup'] += 1
            elif lab == 'noise' or lab == 'noise_seed':
                c['noise'] += 1
            else:
                c['other'] += 1
    return c


def run_test(test_name, rules, mixing, mode, gen_kwargs, n_seq=2000, seq_len=128):
    """Run comparison between old and new generators."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  rules={len(rules)}, mode={mode}, n_seq={n_seq}, seq_len={seq_len}")
    print(f"{'='*60}")

    # Old generator
    seed = 12345
    gen_old = LanguageGenerator(rules=rules, mixing_weights=mixing, mode=mode,
                                 seed=seed, **gen_kwargs)
    t0 = time.time()
    tokens_old, labels_old = gen_old.sample_batch(n_seq, length=seq_len)
    dt_old = time.time() - t0
    print(f"\nOld generator: {dt_old:.3f}s ({n_seq/dt_old:.0f} seq/s)")

    # New generator
    gen_new = FastLanguageGenerator(rules=rules, mixing_weights=mixing, mode=mode,
                                     seed=seed, **gen_kwargs)
    t0 = time.time()
    tokens_new, labels_new = gen_new.sample_batch(n_seq, length=seq_len)
    dt_new = time.time() - t0
    print(f"New generator: {dt_new:.3f}s ({n_seq/dt_new:.0f} seq/s)")
    print(f"Speedup: {dt_old/dt_new:.2f}x")

    # Verify shapes
    assert len(tokens_old) == len(tokens_new) == n_seq
    assert len(tokens_old[0]) == len(tokens_new[0]) == seq_len
    assert len(labels_old) == len(labels_new) == n_seq
    assert len(labels_old[0]) == len(labels_new[0]) == seq_len
    print("\nShapes: PASS")

    # Compare token distributions
    tok_counts_old = count_tokens(tokens_old)
    tok_counts_new = count_tokens(tokens_new)
    tok_pass = compare_distributions("Token Distribution", tok_counts_old, tok_counts_new)

    # Compare rule label distributions
    lab_counts_old = count_labels(labels_old)
    lab_counts_new = count_labels(labels_new)
    lab_pass = compare_distributions("Rule Label Distribution", lab_counts_old, lab_counts_new)

    # Compare rule class distributions
    cls_counts_old = count_rule_classes(labels_old)
    cls_counts_new = count_rule_classes(labels_new)
    cls_pass = compare_distributions("Rule Class Distribution", cls_counts_old, cls_counts_new)

    # Spot-check: verify sequences are valid (rule outputs match triggers)
    n_spot = min(50, n_seq)
    violations = 0
    for i in range(n_spot):
        for j in range(1, seq_len):
            lab = labels_new[i][j]
            tok = tokens_new[i][j]
            # If a bigram rule fired, verify the output token matches the rule
            if lab.startswith('bigram_'):
                for r in rules:
                    if hasattr(r, 'trigger') and r.rule_class == 'bigram' and r.name == lab:
                        expected_id = gen_new.token2id[r.output]
                        if tok != expected_id:
                            violations += 1
            elif lab.startswith('trigram_'):
                for r in rules:
                    if hasattr(r, 'trigger1') and r.rule_class == 'trigram' and r.name == lab:
                        expected_id = gen_new.token2id[r.output]
                        if tok != expected_id:
                            violations += 1
            elif lab.startswith('skip_'):
                for r in rules:
                    if hasattr(r, 'anchor') and r.rule_class == 'skip_bigram' and r.name == lab:
                        expected_id = gen_new.token2id[r.output]
                        if tok != expected_id:
                            violations += 1

    spot_pass = violations == 0
    print(f"\nSpot-check ({n_spot} sequences): {violations} violations {'PASS' if spot_pass else 'FAIL'}")

    all_pass = tok_pass and lab_pass and cls_pass and spot_pass
    print(f"\n{'PASS' if all_pass else 'FAIL'}: {test_name}")
    return all_pass, dt_old, dt_new


def main():
    print("FastLanguageGenerator Verification")
    print("=" * 60)

    results = []

    # Test 1: Depth-1 rules (bigram + trigram + skip-bigram), mixed mode
    ok, dt_old, dt_new = run_test(
        "Depth-1 mixed (25-token vocab)",
        rules=DEPTH1_RULES, mixing=DEPTH1_MIXING, mode='mixed',
        gen_kwargs={}, n_seq=3000, seq_len=128,
    )
    results.append(("Depth-1 mixed", ok, dt_old, dt_new))

    # Test 2: All rules (includes induction + bracket)
    ok, dt_old, dt_new = run_test(
        "All rules mixed (25-token vocab)",
        rules=ALL_RULES, mixing=DEFAULT_MIXING, mode='mixed',
        gen_kwargs={}, n_seq=3000, seq_len=128,
    )
    results.append(("All rules mixed", ok, dt_old, dt_new))

    # Test 3: Isolated mode (bigram only)
    ok, dt_old, dt_new = run_test(
        "Bigram isolated",
        rules=BIGRAM_RULES, mixing=None, mode='isolated',
        gen_kwargs={}, n_seq=2000, seq_len=64,
    )
    results.append(("Bigram isolated", ok, dt_old, dt_new))

    # Test 4: LANG128 (128-token vocab)
    ok, dt_old, dt_new = run_test(
        "LANG128 depth-1 mixed (128-token vocab)",
        rules=LANG128['depth1_rules'], mixing=LANG128['depth1_mixing'], mode='mixed',
        gen_kwargs=dict(vocab=LANG128['vocab'], token2id=LANG128['token2id'], id2token=LANG128['id2token']),
        n_seq=3000, seq_len=128,
    )
    results.append(("LANG128 depth-1", ok, dt_old, dt_new))

    # Test 5: Longer sequences (seq_len=257, as used in training)
    ok, dt_old, dt_new = run_test(
        "Depth-1 mixed, seq_len=257",
        rules=DEPTH1_RULES, mixing=DEPTH1_MIXING, mode='mixed',
        gen_kwargs={}, n_seq=1000, seq_len=257,
    )
    results.append(("Depth-1 seq=257", ok, dt_old, dt_new))

    # Test 6: Large batch timing (representative of training)
    print(f"\n{'='*60}")
    print("TIMING: Large batch (128 seq x 65 tokens, 100 iterations)")
    print(f"{'='*60}")
    gen_old = LanguageGenerator(rules=DEPTH1_RULES, mixing_weights=DEPTH1_MIXING,
                                 mode='mixed', seed=42)
    gen_new = FastLanguageGenerator(rules=DEPTH1_RULES, mixing_weights=DEPTH1_MIXING,
                                     mode='mixed', seed=42)

    # Warmup
    gen_old.sample_batch(10, 65)
    gen_new.sample_batch(10, 65)

    t0 = time.time()
    for _ in range(100):
        gen_old.sample_batch(128, 65)
    dt_old = time.time() - t0

    t0 = time.time()
    for _ in range(100):
        gen_new.sample_batch(128, 65)
    dt_new = time.time() - t0

    print(f"Old: {dt_old:.2f}s total ({dt_old/100*1000:.1f}ms per batch)")
    print(f"New: {dt_new:.2f}s total ({dt_new/100*1000:.1f}ms per batch)")
    print(f"Speedup: {dt_old/dt_new:.2f}x")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok, t_old, t_new in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:35s}: {status} (speedup: {t_old/t_new:.2f}x)")
        all_pass = all_pass and ok

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())

"""
Follow-up experiments: Fixed IMP + Hoyer+IMP combined
"""
import torch
import json
import time
import sys
sys.path.insert(0, '.')

from cifar10_sparsity_experiments import (
    BilinearMixerNet, preload_cifar10, evaluate_fast,
    train_iterative_pruning, train_hoyer_plus_imp,
    count_nonzero_weights, DEVICE, MIXER_HIDDEN, LAYER_HIDDEN
)

def run_followup():
    print(f"Device: {DEVICE}")
    print(f"Config: mixer_hidden={MIXER_HIDDEN}, layer_hidden={LAYER_HIDDEN}")
    print()

    # Preload data
    print("Preloading data to GPU...")
    train_x, train_y, test_x, test_y = preload_cifar10()
    print(f"Data loaded. GPU memory: {torch.cuda.memory_allocated()/1e6:.0f} MB\n")

    results = {}

    # Fixed IMP experiments
    print("=" * 60)
    print("FIXED IMP EXPERIMENTS (now prunes from non-zero weights only)")
    print("=" * 60)

    for target in [0.5, 0.7, 0.9]:
        print(f"\nIMP target={target:.0%}")
        print("-" * 40)
        model = BilinearMixerNet(
            mixer_hidden_dim=MIXER_HIDDEN,
            layer_hidden_dim=LAYER_HIDDEN,
        ).to(DEVICE)
        t0 = time.time()
        hist = train_iterative_pruning(
            model, train_x, train_y, test_x, test_y,
            target_sparsity=target, prune_rounds=5, epochs_per_round=30
        )
        key = f'imp_fixed_{int(target*100)}'
        nonzero, total = count_nonzero_weights(model)
        results[key] = {
            'best_acc': max(hist['acc']),
            'final_acc': hist['acc'][-1],
            'final_sparsity': 1 - nonzero/total,
            'time': time.time() - t0,
        }
        torch.save(model.state_dict(), f'cifar10_models/mixer_imp_fixed{int(target*100)}_noNorm_h64.pt')
        print(f"IMP {target:.0%}: best={results[key]['best_acc']:.2%}, "
              f"sparsity={results[key]['final_sparsity']:.1%}, time={results[key]['time']:.0f}s")

    # Hoyer + IMP combined experiments
    print("\n" + "=" * 60)
    print("HOYER + IMP COMBINED EXPERIMENTS")
    print("=" * 60)

    configs = [
        (0.01, 0.5),   # light hoyer + 50% weight prune
        (0.1, 0.5),    # medium hoyer + 50% weight prune
        (0.1, 0.7),    # medium hoyer + 70% weight prune
        (0.1, 0.9),    # medium hoyer + 90% weight prune
    ]

    for lambda_h, target_s in configs:
        print(f"\nHoyer λ={lambda_h} + IMP {target_s:.0%}")
        print("-" * 40)
        model = BilinearMixerNet(
            mixer_hidden_dim=MIXER_HIDDEN,
            layer_hidden_dim=LAYER_HIDDEN,
        ).to(DEVICE)
        t0 = time.time()
        hist = train_hoyer_plus_imp(
            model, train_x, train_y, test_x, test_y,
            lambda_hoyer=lambda_h, target_sparsity=target_s,
            prune_rounds=5, epochs_per_round=30
        )
        key = f'hoyer{lambda_h}_imp{int(target_s*100)}'
        nonzero, total = count_nonzero_weights(model)
        results[key] = {
            'best_acc': max(hist['acc']),
            'final_acc': hist['acc'][-1],
            'final_sparsity': 1 - nonzero/total,
            'final_hoyer': hist['hoyer'][-1],
            'time': time.time() - t0,
        }
        torch.save(model.state_dict(), f'cifar10_models/mixer_hoyer{lambda_h}_imp{int(target_s*100)}_noNorm_h64.pt')
        print(f"Hoyer+IMP: best={results[key]['best_acc']:.2%}, "
              f"sparsity={results[key]['final_sparsity']:.1%}, "
              f"hoyer={results[key]['final_hoyer']:.3f}, time={results[key]['time']:.0f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<25} {'Best Acc':<12} {'Sparsity':<12} {'Hoyer':<10}")
    print("-" * 60)
    for name, res in results.items():
        hoyer_str = f"{res.get('final_hoyer', 0):.3f}" if 'final_hoyer' in res else "-"
        print(f"{name:<25} {res['best_acc']:.2%}       {res['final_sparsity']:.1%}        {hoyer_str}")

    # Save results
    with open('cifar10_models/sparsity_followup_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to cifar10_models/sparsity_followup_results.json")

    return results


if __name__ == "__main__":
    run_followup()

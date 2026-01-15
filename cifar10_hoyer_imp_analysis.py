# %%
"""
CIFAR-10 Hoyer + IMP Combined Model Analysis

Analyzes the trained Hoyer λ=0.1 + IMP 70% sparse bilinear mixer model.
"""

import torch
from utils_cifar10 import (
    DEVICE, load_model, get_cifar10_test_loader,
    analyze_weight_sparsity, print_weight_sparsity, plot_weight_matrices,
    collect_activations, compute_hoyer_stats, print_hoyer_stats, plot_hoyer_distribution,
    plot_channel_activation_freq, compute_direct_logit_effects, plot_direct_logit_effects,
    plot_sample_analysis, per_class_analysis, print_per_class, plot_per_class,
    compute_d_vector_norms, plot_d_norms, plot_head_cosine_similarity,
    run_single_ablation_study, find_bilinear_matters_samples, run_detailed_ablation,
    print_final_summary
)

print(f"Device: {DEVICE}")

# %%
# Load model and data
MODEL_PATH = 'cifar10_models/mixer_hoyer0.1_imp70_noNorm_h64.pt'
MODEL_NAME = 'Hoyer λ=0.1 + IMP 70%'

model = load_model(MODEL_PATH, mixer_hidden=64, layer_hidden=64, embed_dim=128, n_blocks=1)
test_loader, test_data = get_cifar10_test_loader()

print(f"Loaded model from: {MODEL_PATH}")

# %%
# (1) Weight sparsity analysis
weight_sparsity = analyze_weight_sparsity(model)
overall_sparsity = print_weight_sparsity(weight_sparsity)
plot_weight_matrices(model)

# %%
# (2) Collect activations
print("\nCollecting activations across test set...")
all_mixer_h, all_layer_h, all_preds, all_labels = collect_activations(model, test_loader)
print(f"Mixer activations shape: {all_mixer_h.shape}")
print(f"Layer activations shape: {all_layer_h.shape}")
print(f"Test accuracy: {(all_preds == all_labels).float().mean():.2%}")

# %%
# (3) Hoyer sparsity statistics
hoyer_stats = compute_hoyer_stats(all_mixer_h, all_layer_h)
print_hoyer_stats(hoyer_stats)
plot_hoyer_distribution(hoyer_stats)

# %%
# (4) Channel activation frequency
mixer_freq, layer_freq = plot_channel_activation_freq(all_mixer_h, all_layer_h, threshold=0.1)

# %%
# (5) Direct logit effects
layer_direct_effects = compute_direct_logit_effects(model)
plot_direct_logit_effects(layer_direct_effects)

# %%
# (6) Sample analysis
correct_mask = (all_preds == all_labels)
correct_indices = torch.where(correct_mask)[0]
plot_sample_analysis(model, test_data, correct_indices, layer_direct_effects, n_samples=3)

# %%
# (7) Per-class analysis
per_class = per_class_analysis(all_preds, all_labels, hoyer_stats['hoyer_per_sample'])
print_per_class(per_class)
plot_per_class(per_class)

# %%
# (8) D-vector norms
d_norms = compute_d_vector_norms(model)
plot_d_norms(d_norms)

# %%
# (9) Head cosine similarity
plot_head_cosine_similarity(model)

# %%
# (10) Single sample ablation study
sample_idx = correct_indices[0].item()
run_single_ablation_study(model, test_data, sample_idx)

# %%
# (11) Find samples where bilinear blocks matter
bilinear_matters_indices, embed_only_preds, embed_only_acc = find_bilinear_matters_samples(
    model, test_data, all_preds, all_labels
)

# %%
# (12) Detailed ablation on samples where bilinear blocks matter
run_detailed_ablation(model, test_data, bilinear_matters_indices, all_preds, n_examples=3)

# %%
# Final summary
print_final_summary(MODEL_NAME, all_preds, all_labels, overall_sparsity, hoyer_stats, embed_only_acc)

print()
print("Saved plots:")
print("  - cifar10_images/hoyer_imp_weight_sparsity.png")
print("  - cifar10_images/hoyer_imp_sparsity_dist.png")
print("  - cifar10_images/hoyer_imp_activation_freq.png")
print("  - cifar10_images/hoyer_imp_direct_effects.png")
print("  - cifar10_images/hoyer_imp_sample_analysis.png")
print("  - cifar10_images/hoyer_imp_per_class.png")
print("  - cifar10_images/hoyer_imp_d_norms.png")
print("  - cifar10_images/hoyer_imp_head_cossim.png")
print("  - cifar10_images/hoyer_imp_ablation_single.png")
print("  - cifar10_images/hoyer_imp_ablation_bilinear_matters.png")

# %%

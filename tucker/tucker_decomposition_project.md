# Tucker Decomposition for Bilinear Circuit Interpretability
## Project File for Claude Code

**Status:** Active research — work through stages sequentially.  
**Stack:** PyTorch, existing AlgZoo bilinear infrastructure.  
**Goal:** Determine whether Tucker decomposition + sparse rotation recovers interpretable
circuits from trained bilinear layers, building from synthetic toy cases to the
bigram/trigram language model.

---

## Background and Core Hypothesis

A bilinear layer computes:
```
y = (Lx) ⊙ (Rx)    shape: (d_hidden,) → projected to (d_out,) via W_out
```
As a single weight tensor, this is a 3rd order object:
```
W[i,j,k] = sum_h  L[h,i] * R[h,j] * W_out[k,h]
```
This is exactly a CP decomposition of rank d_hidden. Each hidden unit h contributes
one rank-1 term: the outer product of row h of L, row h of R, and row h of W_out.

**The interpretability question:** Does the trained model use d_hidden independent
rank-1 interactions, or does it use a smaller number of interactions in a higher-
dimensional basis? Tucker decomposition finds this out by compressing to the minimal
subspace and then asking how interactions are structured within that subspace.

**Core hypothesis:** For tasks with clean structure (AND-gates, bigrams, skip-bigrams),
Tucker + sparse rotation will recover the true interaction structure as a sparse core
tensor, where each nonzero entry corresponds to one interpretable computation rule.

---

## Mathematical Setup

### Tucker Decomposition of a Bilinear Layer

Given weight tensor W of shape (d_in, d_in, d_out), Tucker writes:
```
W = G ×_1 A ×_2 B ×_3 C
```
Explicitly:
```
W[i,j,k] = sum_{p,q,r} G[p,q,r] * A[i,p] * B[j,q] * C[k,r]
```
where:
- A: (d_in, r_A) — left input factor matrix, orthogonal columns
- B: (d_in, r_B) — right input factor matrix, orthogonal columns  
- C: (d_out, r_C) — output factor matrix, orthogonal columns
- G: (r_A, r_B, r_C) — core tensor, dense in general

Standard HOSVD gives the unique orthogonal Tucker decomposition. The core G is
generally dense even when W has a simple CP structure — because HOSVD rotates
factors to be orthogonal regardless of the underlying structure.

### The Rotation Problem

A CP tensor with rank r appears as a superdiagonal Tucker core ONLY if the CP
factor vectors happen to be orthogonal. In general, HOSVD rotates the basis and
the core becomes dense. Explicitly:

If W = sum_h a_h ⊗ b_h ⊗ c_h (CP), then Tucker gives:
```
G_HOSVD = G_true ×_1 R_A^T ×_2 R_B^T ×_3 R_C^T
```
where R_A, R_B, R_C are the rotations HOSVD applies to make factors orthogonal.
G_true would be superdiagonal; G_HOSVD is dense.

**The fix:** After HOSVD, find rotations R_A, R_B, R_C ∈ O(r) that solve:
```
min_{R_A, R_B, R_C} || (G ×_1 R_A^T ×_2 R_B^T ×_3 R_C^T) - diag ||_F^2
```
This is the "Diagonalize" step of ODT (Orthogonalize, Diagonalize, Truncate).
The rotated factors are A_rot = A @ R_A, etc. These are no longer guaranteed
orthogonal but represent the interpretable basis.

**There is no closed-form solution for 3rd order tensors.** Use iterative
Jacobi-style sweeps: fix R_B, R_C, optimize R_A via SVD on the unfolded core,
repeat. Converges in practice for low-rank tensors. See PITFALL 1.

### Residual Stream Treatment

Do NOT fold the residual stream into the Tucker decomposition. Treat the total
function as:
```
z = x + W(x, x)    (residual + bilinear contribution)
```
Decompose only W(x,x). The identity path is always present and zero-parameter —
it shows up as a "free" pass-through that you never need to explain. If the
Tucker decomposition of W recovers near-zero core entries, it means the bilinear
layer is doing nothing and the residual is carrying everything. This is a valid
and interpretable outcome. See PITFALL 2.

### Inter-Layer Alignment (Two-Layer Case)

L1 has output factor C_1 (shape d_out × r_C). L2 has input factors A_2, B_2
(shape d_in × r_A, d_in × r_B). After rotating L1's Tucker, the rotated output
factor is C_1_rot = C_1 @ R_C.

Cross-layer sparsity means the overlap matrices:
```
overlap_left  = C_1_rot.T @ A_2_rot    (r_C × r_A)
overlap_right = C_1_rot.T @ B_2_rot    (r_C × r_B)
```
should be sparse. A dense overlap means L2 reads from many L1 features simultaneously.
A sparse overlap means L2 only reads from a few specific L1 output directions.

**Joint optimization objective across two layers:**
```
L = λ_1 * ||G_1_rot - diag||^2     (L1 sparse nodes)
  + λ_2 * ||G_2_rot - diag||^2     (L2 sparse nodes)
  + λ_3 * ||overlap_left||_1       (sparse edges, left)
  + λ_3 * ||overlap_right||_1      (sparse edges, right)
```
λ_3 = 0 gives independent layer-wise ODT. Tuning λ_3 > 0 trades core sparsity
for inter-layer alignment. The optimal λ_3 / λ_1 ratio tells you how much
cross-layer mixing the model actually does.

---

## Ghost Circuits

A nonzero core entry G[p,q,r] represents a *possible* interaction between input
directions A[:,p] and B[:,q] producing output direction C[:,r]. It only fires when
both (A[:,p] · x) and (B[:,q] · x) are simultaneously large.

**For clean synthetic tasks with gradient descent training:** ghost circuits
should be minimal. If two input directions never co-activate in training data,
the gradient signal for that core entry is zero throughout training, so it stays
near initialization (≈ 0 with small init). The weights are approximately
self-cleaning.

**Exception:** superposition. If a Tucker factor direction encodes a mixture of
two features (e.g., "feature_a OR feature_e"), a core entry for that direction
may receive gradient signal from the feature_e co-occurrence even though feature_a
never co-occurs with the paired direction. This creates ghost circuits that look
real in the weights but never fire for the intended feature pair.

**Practical implication for these experiments:** run a quick co-occurrence check
after computing the Tucker decomposition. For each nonzero core entry (p,q,r),
compute the empirical frequency that both (A[:,p] · x) > threshold and
(B[:,q] · x) > threshold simultaneously on the dataset. Zero-frequency entries
are ghosts. Report ghost rate as a diagnostic.

---

## Stage 1: Synthetic CP Tensor — Verify Tucker Recovery

### Setup

Construct a weight tensor as an explicit rank-r CP sum with known factors:
```python
def make_cp_tensor(d_in, d_out, rank, orthogonal=True):
    if orthogonal:
        A = torch.linalg.qr(torch.randn(d_in, rank))[0]  # orthogonal left factors
        B = torch.linalg.qr(torch.randn(d_in, rank))[0]  # orthogonal right factors
    else:
        A = F.normalize(torch.randn(d_in, rank), dim=0)
        B = F.normalize(torch.randn(d_in, rank), dim=0)
    C = torch.randn(d_out, rank)
    # W[i,j,k] = sum_r A[i,r] * B[j,r] * C[k,r]
    W = torch.einsum('ir,jr,kr->ijk', A, B, C)
    return W, A, B, C
```

Run two sub-cases:
- **1A-ortho:** orthogonal CP factors (Tucker should find superdiagonal core exactly)
- **1A-nonortho:** non-orthogonal CP factors (HOSVD gives dense core, rotation needed)

### Pipeline to Implement

```python
def tucker_pipeline(W, rank):
    # Step 1: HOSVD
    G, factors = hosvd(W, rank)  # use tensorly or implement via unfolding SVD
    A_hosvd, B_hosvd, C_hosvd = factors
    
    # Step 2: Iterative Jacobi rotation toward superdiagonality
    R_A, R_B, R_C = find_sparse_rotation(G, n_iters=500)
    
    # Step 3: Apply rotations
    G_rot = rotate_core(G, R_A, R_B, R_C)
    A_rot = A_hosvd @ R_A
    B_rot = B_hosvd @ R_B
    C_rot = C_hosvd @ R_C
    
    return G_rot, A_rot, B_rot, C_rot

def find_sparse_rotation(G, n_iters):
    # Jacobi sweep: fix two rotations, optimize third via SVD of unfolded core
    R_A, R_B, R_C = [torch.eye(G.shape[i]) for i in range(3)]
    for _ in range(n_iters):
        # Optimize R_A: unfold G along mode 0, apply current R_B, R_C
        G_bc = torch.einsum('pqr,qj,rk->pjk', G, R_B, R_C)
        G_unfold = G_bc.reshape(G.shape[0], -1)  # (r_A, r_B*r_C)
        # Target: make G_unfold @ R_A^T superdiagonal-ish
        # Use SVD of the "off-diagonal mass" matrix
        U, S, Vh = torch.linalg.svd(G_unfold @ G_unfold.T)
        R_A = U @ Vh  # rotation that diagonalizes the mode-0 gram matrix
        # Repeat for R_B, R_C
        ...
    return R_A, R_B, R_C
```

**NOTE:** tensorly (pip install tensorly) has HOSVD built in as
`tensorly.decomposition.tucker`. Use it. Don't reimplement HOSVD from scratch.

### Expected Results

| Sub-case | HOSVD core | After rotation | Factor recovery |
|----------|-----------|----------------|-----------------|
| 1A-ortho | Superdiagonal already | No change needed | A_rot ≈ A_true |
| 1A-nonortho | Dense | Becomes approximately superdiagonal | A_rot ≈ A_true up to permutation/sign |

**Metric:** off-diagonal mass fraction = ||G - diag(G)||_F / ||G||_F. Should be
near 0 after rotation for both sub-cases.

**Failure mode to watch for (PITFALL 1):** The Jacobi rotation will converge to a
local minimum if rank > ~10 and the tensor isn't exactly CP. Add random restarts
(~10) and take the rotation with lowest off-diagonal mass. If all restarts give
high off-diagonal mass, the tensor is genuinely Tucker but not CP.

---

## Stage 2: Train Single Bilinear Layer on Known AND-Gate Task

### Task Construction

```python
# 5 boolean input features, 3 AND-gate outputs
# Output 0 = feature_0 AND feature_1
# Output 1 = feature_2 AND feature_3
# Output 2 = feature_0 AND feature_4
# One-hot encoding of all 2^5 = 32 input combinations

def make_and_gate_dataset():
    inputs = torch.tensor(list(itertools.product([0,1], repeat=5))).float()
    targets = torch.stack([
        inputs[:,0] * inputs[:,1],
        inputs[:,2] * inputs[:,3],
        inputs[:,0] * inputs[:,4],
    ], dim=1)
    return inputs, targets
```

### Model

Standard bilinear layer (CP form):
```python
class BilinearLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.L = nn.Linear(d_in, d_hidden, bias=False)
        self.R = nn.Linear(d_in, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_out, bias=False)
    
    def forward(self, x):
        return self.out(self.L(x) * self.R(x))
    
    def weight_tensor(self):
        # Returns W[i,j,k] = sum_h L[h,i]*R[h,j]*out[k,h]
        return torch.einsum('hi,hj,kh->ijk', self.L.weight, self.R.weight, self.out.weight)
```

Use d_hidden = 16, d_in = 5, d_out = 3. Train with MSE loss, Adam, until
convergence. The model has more hidden dimensions than needed (rank-3 task, rank-16
model) — this is intentional to test whether Tucker finds the true rank.

### Analysis

After training:
1. Extract W = model.weight_tensor()
2. Run Tucker pipeline with rank search: try r = 1, 2, 3, 4, 5
3. For each rank, compute reconstruction error and off-diagonal mass after rotation
4. At true rank (r=3), expect: near-zero reconstruction error AND near-superdiagonal core

**Expected core structure at r=3:**
```
G_rot[0,0,0] ≈ s_0  (feature_0 ⊗ feature_1 → output_0)
G_rot[1,1,1] ≈ s_1  (feature_2 ⊗ feature_3 → output_1)
G_rot[2,2,2] ≈ s_2  (feature_0 ⊗ feature_4 → output_2)
all other entries ≈ 0
```
And A_rot columns should align with the true input feature directions (one-hot
vectors e_0, e_1, e_2, e_3, e_4).

**Metric:** cosine similarity between recovered Tucker factor columns and true
feature directions. Should be > 0.99 for clean cases.

### What Outputs 0 and 2 Sharing Feature 0 Looks Like

Outputs 0 and 2 both use feature_0 as their left input. Tucker should detect this:
the left factor A_rot should have feature_0 appear in BOTH component 0 and
component 2 (columns 0 and 2 of A_rot both point toward e_0). This is a sign that
Tucker is revealing shared structure that CP would just repeat.

**Check this explicitly:** compute A_rot[:,0] · A_rot[:,2] (cosine similarity of
left factors for components 0 and 2). Should be high (near 1.0) since both use
feature_0. For components 0 and 1 (which don't share any features), should be
near 0.

**PITFALL 2:** The residual stream issue. If you've added a skip connection
(z = x + W(x,x)), the Tucker decomposition of W alone is correct. Do NOT include
the skip connection weights in W. If your BilinearLayer has no skip connection,
this doesn't apply here — but note it for the two-layer case.

---

## Stage 3: Gated Sparse Tucker as Architecture

### Motivation

Instead of post-hoc fitting Tucker + rotation, train a model where the core G has
explicit binary gates. This prevents the cancellation/ghost-circuit problem
architecturally: a gated-off entry cannot accumulate weight that cancels with
another entry.

### Architecture

```python
class GatedTuckerBilinear(nn.Module):
    def __init__(self, d_in, d_out, r_A, r_B, r_C, gate_temp=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_in, r_A))   # left factor
        self.B = nn.Parameter(torch.randn(d_in, r_B))   # right factor
        self.C = nn.Parameter(torch.randn(d_out, r_C))  # output factor
        self.G = nn.Parameter(torch.randn(r_A, r_B, r_C) * 0.1)  # core
        self.gate_logits = nn.Parameter(torch.zeros(r_A, r_B, r_C))  # gate per entry
        self.temp = gate_temp
    
    def gates(self, hard=False):
        if hard:
            return (self.gate_logits > 0).float()
        return torch.sigmoid(self.gate_logits / self.temp)
    
    def forward(self, x):
        G_eff = self.G * self.gates()
        # Reconstruct W from Tucker factors
        W = torch.einsum('pqr,ip,jq,kr->ijk', G_eff, self.A, self.B, self.C)
        # Apply bilinear
        left = x @ self.A   # (batch, r_A) — wait, this is wrong, see note below
        # Actually: apply full W directly
        return torch.einsum('ijk,bi,bj->bk', W, x, x)
    
    def gate_loss(self):
        return self.gates().sum()  # L0 approximation via sigmoid
```

**NOTE on efficiency:** einsum over full W is expensive for large d_in. For
experiments here (d_in ≤ 32) it's fine. For larger models, use the factored form:
```
left = x @ A        (batch, r_A)
right = x @ B       (batch, r_B)
core_out = einsum('pqr,bp,bq->br', G_eff, left, right)  (batch, r_C)
out = core_out @ C.T   (batch, d_out)
```

### Training

```
L_total = L_task + λ_gate * gate_loss()
```

Anneal temperature: start temp=5.0, decay to 0.1 over training. At end, binarize
gates (hard=True) and fine-tune core G values with frozen gate pattern.

### Expected Results

On the AND-gate task with r_A = r_B = r_C = 5 (slightly overcomplete):
- 3 gates should remain active after training, corresponding to the 3 AND-gates
- Inactive gates should have G entries near zero
- Factor directions should recover the true feature vectors

**Key comparison with Stage 2:** Does the gated Tucker find cleaner factor
directions than post-hoc Tucker + rotation? Does it converge to lower off-diagonal
mass? Report both metrics for direct comparison.

**PITFALL 3:** With small d_in and clean tasks, both methods will likely work.
The gated Tucker advantage should show up in cases with superposition or when
d_hidden >> true rank. Test with d_hidden = 32, true rank = 3 to stress-test.

---

## Stage 4: Two-Layer Stacked Bilinear with Residual

### Task Construction

Compose two layers of AND-gate computation:
```
Layer 1: h_0 = x_0 AND x_1,  h_1 = x_2 AND x_3,  h_2 = x_4 AND x_5
Layer 2: z_0 = h_0 AND h_1,  z_1 = h_1 AND h_2
```
Input: 6 boolean features. Output: 2 boolean values.
The computation requires L2 to use L1's outputs — genuine cross-layer composition.

With residual streams:
```
r_0 = x
r_1 = x + L1(x, x)      # L1 output added to residual
z = r_1 + L2(r_1, r_1)  # L2 reads enriched residual
```

L2 now sees both the original x and the L1 features h_0, h_1, h_2. Its AND-gate
computation (h_0 AND h_1) should focus on the h directions, not the x directions.
But the residual means x directions are also present — Tucker for L2 should find
near-zero weights on the x-direction components.

### Analysis Pipeline

```python
# Step 1: Tucker for L1 (ignoring residual — just decompose L1's weight tensor)
W1 = model.layer1.weight_tensor()
G1_rot, A1_rot, B1_rot, C1_rot = tucker_pipeline(W1, rank=3)

# Step 2: Tucker for L2 (same — just L2's weight tensor)
W2 = model.layer2.weight_tensor()
G2_rot, A2_rot, B2_rot, C2_rot = tucker_pipeline(W2, rank=2)

# Step 3: Inter-layer alignment
overlap_left  = C1_rot.T @ A2_rot   # (r_C1, r_A2)
overlap_right = C1_rot.T @ B2_rot   # (r_C1, r_B2)

# Step 4: Joint rotation optimization (optional, try first without)
# Optimize R_C1, R_A2, R_B2 jointly to maximize sparsity of overlaps
# while maintaining superdiagonality of G1_rot and G2_rot

# Step 5: Ghost circuit check
for each nonzero (p,q,r) in G2_rot:
    compute P(A2_rot[:,p]·residual > thresh AND B2_rot[:,q]·residual > thresh)
    over dataset
```

### Expected Results

**Overlap matrices:** should be approximately block-diagonal or sparse. Specifically,
C1_rot's 3 columns (L1's h_0, h_1, h_2 output directions) should align with 2-3
specific columns of A2_rot and B2_rot (the L2 input directions that read h values).

**L2 core G2_rot:** should have 2 nonzero entries: (h_0, h_1, z_0) and (h_1, h_2, z_1).

**Residual contamination:** L2's Tucker may find some weight on x-direction components
(via the residual). These should be low-magnitude and correspond to ghost circuits
(zero co-occurrence in data if x_i never satisfy the L2 AND condition directly).

**PITFALL 4 (Critical):** Layer-wise Tucker optimizes each layer's core independently.
The rotation that superdiagonalizes G1 may be completely different from the rotation
that aligns C1 with A2. If you run layer-wise ODT naively, C1_rot and A2_rot will
be in different bases and the overlap matrix will look dense even if the true
inter-layer connectivity is sparse. 

Fix: after layer-wise ODT, add a second pass that jointly rotates C1_rot and A2_rot
to maximize sparsity of their overlap. This is a matrix sparsification problem
solvable by iterative column-pair rotations (Jacobi on the overlap matrix).

---

## Stage 5: Bigram/Trigram Language Model

### Model Description

Assumed: existing trained model from AlgZoo/bigram-trigram corpus. Known data
generating process with interpretable rules. Synthetic vocabulary with ~50 tokens.

### Analysis Plan

Apply the full pipeline from Stages 1-4:

1. **Extract weight tensors** for each bilinear layer (and bilinear attention heads
   if present — see Attention Extension below).

2. **Tucker + rotation per layer** with rank search. Use reconstruction error
   elbow to select rank automatically:
   ```python
   errors = [tucker_reconstruction_error(W, rank=r) for r in range(1, d_hidden+1)]
   # Select rank at elbow of error curve
   ```

3. **Embed factors in vocab space.** For the first layer, multiply factor matrices
   by the embedding matrix E:
   ```
   A_vocab = E @ A_rot    (V × r_A) — left factors in token space
   B_vocab = E @ B_rot    (V × r_B) — right factors in token space
   C_vocab = C_rot @ U.T  (r_C × V) — output factors in logit space (U = unembedding)
   ```
   Now each column of A_vocab is a distribution over tokens — the "left input
   feature" in interpretable terms. A column that concentrates on NOUN tokens means
   "this feature is activated by NOUN tokens in the left position."

4. **Read off circuit rules from core.** For each nonzero G_rot[p,q,r]:
   - Left feature: top tokens in A_vocab[:,p]
   - Right feature: top tokens in B_vocab[:,q]  
   - Output effect: top logit changes in C_vocab[r,:]
   - Rule: "when left context is {top tokens in p} and right context is {top tokens
     in q}, boost {top tokens in r}"

5. **Cross-reference with known data generating process.** For each recovered rule,
   check whether it matches a known bigram, trigram, or skip-bigram rule. Report:
   - Precision: fraction of recovered rules that match known rules
   - Recall: fraction of known rules that are recovered
   - Any unexpected rules (potential evidence of learned generalizations)

6. **Ghost circuit check** on the full corpus.

### Expected Outcome

For a model trained on a clean bigram/trigram corpus, expect:
- Small Tucker rank (much less than d_hidden)
- Core entries that read off directly as grammar rules
- Near-zero ghost rate (the model only learned rules that occur in the data)

**The acid test:** Can you read off the grammar from the Tucker cores *without
looking at the data*? If precision and recall are both > 0.9, Tucker decomposition
is working as a weight-only interpretability method for this class of model.

---

## Attention Extension (Bilinear Attention)

For a bilinear attention layer the score is:
```
score(q,k) = [(e_q @ W_Q1^T) · (e_k @ W_K1^T)] × [(e_q @ W_Q2^T) · (e_k @ W_K2^T)]
```

Embed in vocab space:
```
QK1[q,k] = E @ W_Q1^T @ W_K1 @ E^T / sqrt(D)    (V × V)
QK2[q,k] = E @ W_Q2^T @ W_K2 @ E^T / sqrt(D)    (V × V)
OV[k,o]  = E @ W_V^T @ W_O^T @ U^T              (V × V)
```

Full 3rd order tensor:
```
T[q,k,o] = QK1[q,k] × QK2[q,k] × OV[k,o]
```

Tucker decomposition of T gives the same structure as the MLP case — factor
matrices in query, key, and output token space, with a core tensor describing
which (query-feature, key-feature, output-feature) interactions exist.

**PITFALL 5:** RoPE or learned positional embeddings contaminate the QK matrices.
E @ W_Q1^T @ W_K1 @ E^T assumes pure content-based attention. With positional
encoding, the actual QK matrix is position-dependent and can't be embedded in vocab
space without averaging or conditioning on position. For the bigram/trigram model,
check whether positional encoding is used. If yes, analyze content and position
components separately or condition on specific relative positions.

---

## Implementation Checklist

- [ ] Install tensorly: `pip install tensorly`
- [ ] Implement `tucker_pipeline(W, rank)` with HOSVD + Jacobi rotation
- [ ] Implement `ghost_circuit_check(G_rot, A_rot, B_rot, dataset)`
- [ ] Stage 1A-ortho: synthetic CP tensor, verify superdiagonal recovery
- [ ] Stage 1A-nonortho: non-orthogonal CP, verify rotation works
- [ ] Stage 2: AND-gate task, single bilinear layer
- [ ] Stage 2 extension: verify shared-feature detection (outputs 0 and 2 sharing feature_0)
- [ ] Stage 3: Gated Tucker architecture, compare to post-hoc Stage 2
- [ ] Stage 4: Two-layer stacked bilinear, test inter-layer alignment
- [ ] Stage 4 PITFALL 4: second-pass joint rotation of C1/A2 overlap
- [ ] Stage 5: Bigram/trigram model, precision/recall vs known grammar

---

## Key Pitfall Summary

| # | Pitfall | Fix |
|---|---------|-----|
| 1 | Jacobi rotation gets stuck in local minimum | 10+ random restarts, take best |
| 2 | Including residual stream in W | Decompose only bilinear W, treat identity separately |
| 3 | Gated Tucker advantage invisible on easy tasks | Test with d_hidden >> true rank |
| 4 | Layer-wise ODT bases don't align across layers | Second-pass joint rotation of inter-layer overlap |
| 5 | Positional encoding contaminates QK embedding | Separate content and position or condition on relative position |

---

## Success Criteria

The project succeeds if, on the bigram/trigram model:
1. Tucker rank << d_hidden (compression is real)
2. Core entries read as interpretable grammar rules without data
3. Precision and recall vs known grammar both > 0.9
4. Ghost circuit rate < 0.1

If these aren't met, the most likely explanations in order:
- Superposition: model is packing multiple features per hidden dimension →
  Tucker basis won't be interpretable without sparse dictionary step
- Inter-layer alignment failure: layer-wise ODT produces incompatible bases →
  need joint optimization
- Task too complex for weight-only analysis → need data-informed basis (PCA on
  activations first, Tucker second)

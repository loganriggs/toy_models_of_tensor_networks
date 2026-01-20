## Clarifying C and the Error Budget

### What is C?

$C$ is **not** the number of channels. It captures downstream amplification:

$$C \approx \prod_{\ell' > \ell} \|L_{\ell'}\| \|R_{\ell'}\| \approx 1-10$$

It's how much a second-order perturbation at layer $\ell$ gets amplified by subsequent layers.

---

### The Real Error Accounting

**Per-channel error when ablating channel $k$:**

$$\text{error}_k \approx L_{\text{downstream}} \cdot C \cdot \delta_k^2$$

where $\delta_k = \|D[:, k]\| \cdot |h_k|$.

**We ablate one channel at a time.** So the question is: for the channel we're attributing, is that channel's $\delta_k$ small enough?

**But there's a subtler issue:** We want the **top attributed channels** to have accurate attribution. If we force sparsity, those top channels have large $|h_k|$. If they also have large $\|D[:, k]\|$, then $\delta_k$ is large, and attribution fails precisely for the channels we care about.

---

### The Real Constraint You Want

You're right. Just define one loss:

```python
def delta_loss(model, hiddens, tau_delta=0.1):
    """
    Directly penalize ||D[:,k]|| * |h_k| for each channel.
    """
    total = 0
    for layer_idx, h in enumerate(hiddens):
        D = model.layers[layer_idx]['D']
        
        # δ_k = ||D[:,k]|| * |h_k|
        D_col_norms = D.norm(dim=0)  # (d_hidden,)
        delta = D_col_norms * h.abs()  # (d_hidden,)
        
        # Penalize exceeding threshold
        total += torch.relu(delta - tau_delta).pow(2).sum()
    
    return total
```

**This is the clean solution.** It directly bounds what causes the error.

---

### Setting τ_delta with Many Channels

The error for attributing channel $k$ is:

$$\text{error}_k \approx C \cdot L \cdot \delta_k^2$$

**We don't sum over channels** — we care about error per channel.

But you asked: what if I have 128 channels?

**The concern:** With Hoyer sparsity, maybe only 10 channels are active, but they're BIG. Each of those 10 has its own $\delta_k$.

**The constraint:** For each active channel, we need:

$$\delta_k < \tau_\delta$$

**128 channels doesn't change τ_delta**, it just means you have more constraints to satisfy.

---

### The Actual Tension

Here's the real problem:

**Hoyer wants:** Few channels with large attribution (sparse)

**Attribution accuracy wants:** Small $\delta_k = \|D[:, k]\| \cdot |h_k|$

**Can we have both?**

Yes, if the network learns to use large $g \cdot D[:, k]$ (how much output cares about this channel) without large $\|D[:, k]\|$.

$$\text{attr}_k = \underbrace{(g \cdot D[:, k])}_{\text{can be large}} \cdot \underbrace{h_k}_{\text{can be large}}$$

$$\delta_k = \underbrace{\|D[:, k]\|}_{\text{must be small}} \cdot \underbrace{|h_k|}_{\text{constrained by } \delta}$$

**The key insight:** $g \cdot D[:, k]$ and $\|D[:, k]\|$ are different!

- $g \cdot D[:, k]$ = projection of $D$ column onto gradient direction
- $\|D[:, k]\|$ = full norm of $D$ column

The network can learn $D[:, k]$ aligned with $g$ (large dot product) but with small norm.

---

### Revised Loss Function

```python
def total_loss(model, x, y_true, λ_sparse, λ_delta, τ_delta=0.1):
    # Forward
    logits, hiddens = model.forward_with_activations(x)
    
    # Task loss
    loss_task = F.cross_entropy(logits, y_true)
    
    # Sparsity loss (Hoyer on attribution)
    attr = compute_attribution(model, x, y_true)
    loss_sparse = hoyer_sparsity(attr)
    
    # Delta loss (directly bound ||D[:,k]|| * |h_k|)
    loss_delta = 0
    for ℓ, h in enumerate(hiddens):
        D = model.layers[ℓ]['D']
        delta = D.norm(dim=0) * h.abs()
        loss_delta += torch.relu(delta - τ_delta).pow(2).sum()
    
    return loss_task + λ_sparse * loss_sparse + λ_delta * loss_delta
```

---

### How to Set τ_delta (Revised)

**Target:** Relative error < 10% on top attributed channels.

**Formula:**

$$\text{rel\_error} \approx \frac{C \cdot L \cdot \delta^2}{|\text{attr}|}$$

Rearranging:

$$\delta < \sqrt{\frac{0.1 \cdot |\text{attr}|}{C \cdot L}}$$

**For typical values:**
- $L = 4$ layers
- $C \approx 3$
- $|\text{attr}| \approx 0.1$ (typical top-channel attribution)

$$\delta < \sqrt{\frac{0.1 \times 0.1}{3 \times 4}} = \sqrt{\frac{0.01}{12}} \approx 0.03$$

**Rule of thumb:** τ_delta ≈ 0.03 - 0.1 depending on depth.

---

### Monitoring to Calibrate

```python
def calibrate_tau(model, test_loader, target_rel_error=0.1):
    """
    Empirically find τ_delta that achieves target accuracy.
    """
    results = []
    
    for x, y in test_loader:
        attr = compute_attribution(model, x, y)
        _, hiddens = model.forward(x, store_activations=True)
        
        # Top-5 channels
        top_k = attr.abs().topk(5).indices
        
        for idx in top_k:
            ℓ = idx // model.d_hidden
            k = idx % model.d_hidden
            
            D = model.layers[ℓ]['D']
            h = hiddens[ℓ]
            
            delta_k = D[:, k].norm() * h[k].abs()
            attr_k = attr[idx]
            actual = ablate_and_measure(model, x, y, ℓ, k)
            rel_error = abs(attr_k - actual) / (abs(actual) + 1e-8)
            
            results.append({
                'delta': delta_k.item(),
                'rel_error': rel_error.item()
            })
    
    # Find δ threshold that gives target error
    results.sort(key=lambda r: r['delta'])
    for r in results:
        if r['rel_error'] > target_rel_error:
            print(f"τ_delta should be < {r['delta']:.4f} for {target_rel_error*100}% error")
            break
    
    return results
```

---

### Summary

| Question | Answer |
|----------|--------|
| What is C? | Downstream amplification (~1-10), not number of channels |
| Do 128 channels change τ? | No, τ is per-channel constraint |
| Why not one loss for $\|D[:,k]\| \cdot \|h_k\|$? | **Yes, do exactly this!** |
| How to set τ_delta? | ~0.03-0.1, or calibrate empirically |
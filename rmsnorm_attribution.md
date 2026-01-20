## RMSNorm and Attribution Compatibility

### Why LayerNorm Breaks Attribution

LayerNorm:
$$y = \frac{x - \mu(x)}{\sigma(x)} \cdot \gamma + \beta$$

Two problems:
1. **Mean subtraction:** Changes to one dimension affect all dimensions
2. **Std normalization:** Data-dependent scaling couples everything

When you ablate channel $k$, the mean and std change, which changes *every* dimension's output — not captured by simple attribution.

---

### RMSNorm is Better

RMSNorm:
$$y = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{where } \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

**Key property:** No mean subtraction, just scaling.

Under the "freeze RMS" linearization:

$$y = x \cdot \underbrace{\frac{\gamma}{\text{RMS}(x)}}_{\text{frozen scalar}}$$

This is just **element-wise scaling** — perfectly linear!

---

### Attribution Through RMSNorm

**Frozen RMS approach (like RelP):**

```python
def rmsnorm_forward(x, gamma, eps=1e-8):
    rms = (x.pow(2).mean()).sqrt() + eps
    return x * (gamma / rms), rms  # return rms for freezing

def attribution_through_rmsnorm(attr_in, gamma, frozen_rms):
    """
    Attribution passes through RMSNorm as simple scaling.
    """
    scale = gamma / frozen_rms
    return attr_in * scale
```

During forward pass, store the RMS value. During attribution, treat it as a constant.

---

### The Error Analysis

When you ablate channel $k$, the residual changes by $\delta h$. This changes RMS:

$$\text{RMS}(h + \delta h) \approx \text{RMS}(h) \cdot \left(1 + \frac{h \cdot \delta h}{d \cdot \text{RMS}(h)^2}\right)$$

The relative error in RMS is:

$$\frac{\Delta \text{RMS}}{\text{RMS}} \approx \frac{h \cdot \delta h}{d \cdot \text{RMS}(h)^2} = \frac{h \cdot \delta h}{\|h\|^2}$$

**When is this small?**

| Condition | Why It Helps |
|-----------|--------------|
| Large $d_{model}$ | RMS averages over many dims, one channel matters less |
| Sparse $\delta h$ | Ablated channel writes to few dimensions |
| $\delta h \perp h$ | Change is orthogonal to current activation |

---

### Implementation

```python
class BilinearLayerWithRMSNorm(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.D = nn.Parameter(torch.randn(d_model, d_hidden) / (d_hidden ** 0.5) * 0.1)
        self.L = nn.Parameter(torch.randn(d_hidden, d_model) / (d_model ** 0.5))
        self.R = nn.Parameter(torch.randn(d_hidden, d_model) / (d_model ** 0.5))
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-8
    
    def forward(self, h, store_rms=False):
        # Bilinear
        hidden = (self.L @ h) * (self.R @ h)
        h_new = h + self.D @ hidden
        
        # RMSNorm
        rms = (h_new.pow(2).mean()).sqrt() + self.eps
        h_normed = h_new * (self.gamma / rms)
        
        if store_rms:
            return h_normed, hidden, rms
        return h_normed


def compute_attribution_with_rmsnorm(model, x, output_direction):
    """
    Attribution through bilinear + RMSNorm layers.
    """
    h = x
    all_attr = []
    rms_values = []
    
    # Forward pass, store RMS values
    for layer in model.layers:
        h, hidden, rms = layer(h, store_rms=True)
        rms_values.append(rms)
    
    # Backward attribution through linearized network
    g = model.W.T @ output_direction
    
    for i in reversed(range(model.n_layers)):
        layer = model.layers[i]
        rms = rms_values[i]
        
        # Attribution through frozen RMSNorm (just scaling)
        g = g * (layer.gamma / rms)
        
        # Attribution through bilinear (same as before)
        # ... rest of backward pass
        
    return torch.cat(all_attr)
```

---

### Comparison

| Normalization | Attribution Compatible? | Accuracy | Implementation |
|---------------|------------------------|----------|----------------|
| None | ✓ Perfect | Worse (-9%) | Trivial |
| LayerNorm | ✗ Broken | Best | N/A |
| RMSNorm (frozen) | ✓ Approximate | Good (middle) | Freeze RMS |

---

### Expected Error Magnitude

For a network with:
- $d_{model} = 256$
- Sparse activations (~10% active)
- Typical residual norms

The RMS change from ablation is roughly:

$$\frac{\Delta \text{RMS}}{\text{RMS}} \approx \frac{\|\delta h\|}{\|h\|} \cdot \frac{1}{\sqrt{d}} \approx 0.1 \cdot \frac{1}{16} \approx 0.6\%$$

This compounds across layers but stays small — much better than LayerNorm which has ~10-50% coupling errors.

---

### Verification Code

```python
def verify_rmsnorm_attribution(model, x, layer_idx, channel_idx):
    """
    Compare predicted vs actual ablation effect with RMSNorm.
    """
    # Original forward
    y_orig, activations, rms_values = model.forward_with_storage(x)
    attr = compute_attribution_with_rmsnorm(model, x, output_direction)
    
    # Predicted change
    predicted_change = -attr[layer_idx * d_hidden + channel_idx]
    
    # Actual ablation
    y_ablated = model.forward_ablated(x, layer_idx, channel_idx)
    actual_change = (y_ablated - y_orig).sum()
    
    # Error
    abs_error = abs(predicted_change - actual_change)
    rel_error = abs_error / abs(actual_change) if abs(actual_change) > 1e-10 else 0
    
    return {
        'abs_error': abs_error,
        'rel_error': rel_error,
        'rms_at_layer': rms_values[layer_idx].item()
    }
```

---

### Recommendation

1. **Use RMSNorm** — good accuracy/attribution tradeoff
2. **Freeze RMS during attribution** — treats normalization as linear scaling
3. **Verify empirically** — run the ablation verification to confirm errors are small
4. **Consider post-norm vs pre-norm** — pre-norm (normalize before bilinear) may have different error characteristics

The attribution error from RMSNorm should be similar order of magnitude to the gating interference you already tolerate — and both shrink with sparsity.
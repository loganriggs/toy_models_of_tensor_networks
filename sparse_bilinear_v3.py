import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =========================
#  Model
# =========================

class BilinearResBlock(nn.Module):
    """
    x_next = x + D( (Lx) ⊙ (Rx) )
    L: [hidden, in]
    R: [hidden, in]
    D: [in, hidden]
    """
    def __init__(self, in_dim, hidden_dim=1, bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.L = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.R = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.D = nn.Linear(hidden_dim, in_dim, bias=bias)
        for m in (self.L, self.R, self.D):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        u = self.L(x)          # [B,H]
        v = self.R(x)          # [B,H]
        h = u * v              # [B,H]
        delta = self.D(h)      # [B,D]
        return x + delta, dict(u=u, v=v, h=h, delta=delta)


class BilinearResNetMNIST(nn.Module):
    """
    Simple bilinear residual network on flattened MNIST.
    """
    def __init__(self, in_dim=784, hidden_dim=1, n_blocks=1, out_dim=10, bias=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            BilinearResBlock(in_dim, hidden_dim, bias=bias)
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        caches = []
        for blk in self.blocks:
            x, c = blk(x)
            caches.append(c)
        logits = self.head(x)
        return logits, caches


def model_device(model):
    return next(model.parameters()).device


# =========================
#  Forward with stream states
# =========================

def forward_stream_states(model, x):
    """
    Forward pass with explicit residual stream states.

    Returns:
      logits:  [B,C]
      x_list:  [x0, x1, ..., xL] each [B,D]
      cache_list: list of per-block dicts with keys 'u','v','h','delta'
    """
    dev = model_device(model)
    x = x.to(dev)
    if x.dim() == 4:
        x = x.view(x.size(0), -1)

    x_list = [x]
    cache_list = []

    x_in = x
    for blk in model.blocks:
        x_out, c = blk(x_in)
        cache_list.append(c)
        x_list.append(x_out)
        x_in = x_out

    logits = model.head(x_list[-1])
    return logits, x_list, cache_list


# =========================
#  One-block exact attribution sanity
# =========================

@torch.no_grad()
def predicted_drop_batch_one_block(model: BilinearResNetMNIST, x, class_idx, block_idx=0):
    """
    Closed-form predicted drop in a 1-block model when zeroing one weight.
    Returns:
      pred_L: [B,H,D]
      pred_R: [B,H,D]
      pred_D: [B,D,H]
    """
    dev = model_device(model)
    x = x.to(dev)
    if x.dim() == 4:
        x = x.view(x.size(0), -1)

    B, D_in = x.shape
    blk = model.blocks[block_idx]

    if isinstance(class_idx, int):
        class_idx = torch.full((B,), class_idx, device=dev, dtype=torch.long)
    else:
        class_idx = class_idx.to(dev)

    w = model.head.weight[class_idx]  # [B,D]

    u = blk.L(x)  # [B,H]
    v = blk.R(x)  # [B,H]
    Dmat = blk.D.weight  # [D,H]
    D_eff = torch.einsum("bd,dh->bh", w, Dmat)  # [B,H]

    WL = blk.L.weight  # [H,D]
    WR = blk.R.weight

    pred_L = D_eff.unsqueeze(-1) * v.unsqueeze(-1) * (WL.unsqueeze(0) * x.unsqueeze(1))
    pred_R = D_eff.unsqueeze(-1) * u.unsqueeze(-1) * (WR.unsqueeze(0) * x.unsqueeze(1))

    h = u * v
    pred_D = (w.unsqueeze(-1) * Dmat.unsqueeze(0)) * h.unsqueeze(1)  # [B,D,H]

    return pred_L, pred_R, pred_D


@torch.no_grad()
def actual_drop_single_weight(model, x, class_idx, block_idx, param_name, idx):
    dev = model_device(model)
    x = x.to(dev)

    logits0, _ = model.forward(x)
    logit0 = logits0[0, class_idx].item()

    if param_name == "head":
        W = model.head.weight
    else:
        blk = model.blocks[block_idx]
        W = getattr(blk, param_name).weight

    old = W[idx].clone()
    W[idx] = 0.0
    logits1, _ = model.forward(x)
    logit1 = logits1[0, class_idx].item()
    W[idx] = old

    return logit0 - logit1


@torch.no_grad()
def assert_predicted_matches_ablation_one_block(model, x, class_idx=0, n_checks=20, tol=1e-5):
    """
    Sanity: in a 1-block model, exact predicted-drop formula matches ablation.
    """
    assert len(model.blocks) == 1, "Assert is for 1-block models."

    dev = model_device(model)
    x = x.to(dev)
    if x.dim() == 4:
        x = x.view(x.size(0), -1)

    x1 = x[0:1]
    pred_L, pred_R, pred_D = predicted_drop_batch_one_block(model, x1, class_idx, 0)

    blk = model.blocks[0]
    H, D_in = blk.L.weight.shape

    g = torch.Generator().manual_seed(0)

    for _ in range(n_checks):
        which = int(torch.randint(0, 3, (1,), generator=g).item())
        if which == 0:
            k = int(torch.randint(0, H, (1,), generator=g).item())
            i = int(torch.randint(0, D_in, (1,), generator=g).item())
            pred = pred_L[0, k, i].item()
            act = actual_drop_single_weight(model, x1, class_idx, 0, "L", (k, i))
        elif which == 1:
            k = int(torch.randint(0, H, (1,), generator=g).item())
            i = int(torch.randint(0, D_in, (1,), generator=g).item())
            pred = pred_R[0, k, i].item()
            act = actual_drop_single_weight(model, x1, class_idx, 0, "R", (k, i))
        else:
            j = int(torch.randint(0, D_in, (1,), generator=g).item())
            k = int(torch.randint(0, H, (1,), generator=g).item())
            pred = pred_D[0, j, k].item()
            act = actual_drop_single_weight(model, x1, class_idx, 0, "D", (j, k))

        if abs(pred - act) > tol:
            raise AssertionError(f"Mismatch pred={pred:.8f} act={act:.8f}")

    print(f"OK: {n_checks} predicted-drop checks matched ablation within {tol}.")


# =========================
#  Conservative attribution (multi-block) + flattening
# =========================

@torch.no_grad()
def kl_divergence_logits(logits_p, logits_q):
    p = F.log_softmax(logits_p, dim=-1)
    q = F.log_softmax(logits_q, dim=-1)
    P = p.exp()
    return (P * (p - q)).sum(dim=-1)


def forward_with_attrib_batch(model: BilinearResNetMNIST, x, class_idx,
                              split_mode="LR", eps=1e-12):
    """
    Conservative block-level attribution for chosen class logit.

    Returns:
      logits:          [B,C]
      attrib_blocks:   list length L
        each entry: dict(L=[B,H,D], R=[B,H,D], D=[B,D,H], logit_block=[B])
    """
    dev = model_device(model)
    x = x.to(dev)

    logits, _ = model.forward(x)
    if x.dim() == 4:
        x_flat = x.view(x.size(0), -1)
    else:
        x_flat = x

    B = x_flat.size(0)

    if isinstance(class_idx, int):
        class_idx = torch.full((B,), class_idx, device=dev, dtype=torch.long)
    else:
        class_idx = class_idx.to(dev)

    W_head = model.head.weight            # [C,D]
    w = W_head[class_idx]                 # [B,D]

    attrib_blocks = []
    x_in = x_flat

    for blk in model.blocks:
        x_out, c = blk(x_in)
        u, v, h, delta = c['u'], c['v'], c['h'], c['delta']  # [B,H], [B,H], [B,H], [B,D]

        # direct contribution of this block to chosen logit
        logit_block = (w * delta).sum(dim=-1)  # [B]

        # fold D and head together
        Dmat = blk.D.weight  # [D,H]
        D_eff = torch.einsum("bd,dh->bh", w, Dmat)  # [B,H]
        c_k = D_eff * u * v  # [B,H]  contribution per hidden channel

        if split_mode == "LR":
            credit_u = 0.5 * c_k
            credit_v = 0.5 * c_k
            credit_Deff = torch.zeros_like(c_k)
        elif split_mode == "LDR":
            credit_u = (1/3) * c_k
            credit_v = (1/3) * c_k
            credit_Deff = (1/3) * c_k
        else:
            raise ValueError("split_mode must be 'LR' or 'LDR'")

        WL = blk.L.weight  # [H,D]
        WR = blk.R.weight

        numer_L = WL.unsqueeze(0) * x_in.unsqueeze(1)  # [B,H,D]
        numer_R = WR.unsqueeze(0) * x_in.unsqueeze(1)

        aL = credit_u.unsqueeze(-1) * numer_L / (u.unsqueeze(-1) + eps)
        aR = credit_v.unsqueeze(-1) * numer_R / (v.unsqueeze(-1) + eps)

        aD = torch.zeros((B, x_in.size(1), blk.hidden_dim),
                         device=dev, dtype=Dmat.dtype)
        if credit_Deff.abs().sum() > 0:
            wd = w.unsqueeze(-1) * Dmat.unsqueeze(0)              # [B,D,H]
            denom = D_eff.unsqueeze(1) + eps                      # [B,1,H]
            aD = credit_Deff.unsqueeze(1) * wd / denom            # [B,D,H]

        attrib_blocks.append(dict(L=aL, R=aR, D=aD, logit_block=logit_block))
        x_in = x_out

    return logits, attrib_blocks


def flatten_one_sample_attrib(model, attrib_blocks, sample_idx=0):
    """
    Flatten L/R/D attributions for a single sample into a length-P vector.
    Order:
      for each block: L (H,D) row-major, R (H,D), D (D,H).
    """
    flat = []
    for b in attrib_blocks:
        flat.append(b["L"][sample_idx].reshape(-1))
        flat.append(b["R"][sample_idx].reshape(-1))
        flat.append(b["D"][sample_idx].reshape(-1))
    return torch.cat(flat, dim=0)


def build_flat_index_map(model):
    """
    Flatten index map matching flatten_one_sample_attrib order.
    Each entry: (param_name, block_idx, idx)
      param_name in {"L","R","D"}, idx is 2D index.
    """
    idx_map = []
    for bi, blk in enumerate(model.blocks):
        H, D_in = blk.L.weight.shape
        D_out, H2 = blk.D.weight.shape
        assert H == H2

        # L
        for k in range(H):
            for i in range(D_in):
                idx_map.append(("L", bi, (k, i)))

        # R
        for k in range(H):
            for i in range(D_in):
                idx_map.append(("R", bi, (k, i)))

        # D
        for j in range(D_out):
            for k in range(H):
                idx_map.append(("D", bi, (j, k)))
    return idx_map


def unflatten_mask_to_struct(model, flat_mask):
    """
    Inverse of flatten_one_sample_attrib for masks.
    """
    masks = []
    offset = 0
    for blk in model.blocks:
        H, D_in = blk.L.weight.shape
        D_out, H2 = blk.D.weight.shape
        assert H == H2

        nL = H * D_in
        nR = H * D_in
        nD = D_out * H

        mL = flat_mask[offset:offset+nL].view(H, D_in); offset += nL
        mR = flat_mask[offset:offset+nR].view(H, D_in); offset += nR
        mD = flat_mask[offset:offset+nD].view(D_out, H); offset += nD

        masks.append({"L": mL, "R": mR, "D": mD})
    return masks


@torch.no_grad()
def apply_bilinear_masks_inplace(model, masks):
    """
    Multiply L/R/D weights by 0/1 masks (in-place).
    """
    for bi, blk_masks in enumerate(masks):
        blk = model.blocks[bi]
        blk.L.weight *= blk_masks["L"].to(blk.L.weight.dtype)
        blk.R.weight *= blk_masks["R"].to(blk.R.weight.dtype)
        blk.D.weight *= blk_masks["D"].to(blk.D.weight.dtype)


@torch.no_grad()
def mask_quality_checks(model, x1, keep_frac=0.99):
    """
    Sanity check: for a single sample,
      * build mask from conservative attribution
      * apply it and measure logit drop + KL.
    """
    dev = model_device(model)
    x1 = x1.to(dev)

    logits0, _ = model.forward(x1)
    pred = int(logits0.argmax(dim=-1).item())

    _, attrib_blocks = forward_with_attrib_batch(model, x1, class_idx=pred, split_mode="LR")
    flat = flatten_one_sample_attrib(model, attrib_blocks, 0)

    # build mask that keeps keep_frac of |attrib| mass
    abs_a = flat.abs()
    vals, idx = torch.sort(abs_a, descending=True)
    csum = torch.cumsum(vals, 0)
    total = vals.sum() + 1e-12
    cutoff = keep_frac * total
    keep_n = int((csum <= cutoff).sum().item())
    keep_n = max(1, keep_n)

    mask_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_flat[idx[:keep_n]] = True
    masks_struct = unflatten_mask_to_struct(model, mask_flat)

    sd = {k: v.clone() for k, v in model.state_dict().items()}
    apply_bilinear_masks_inplace(model, masks_struct)

    logits1, _ = model.forward(x1)
    model.load_state_dict(sd)

    logit_drop = (logits0[0, pred] - logits1[0, pred]).item()
    kl = kl_divergence_logits(logits0, logits1)[0].item()

    return pred, logit_drop, kl


# =========================
#  Gradient-folded channel scores (batch-averaged)
# =========================
def channel_attrib_batch_grad_folded(model, x, y=None, class_source="pred"):
    """
    Per-sample grad-folded channel attributions.

    Returns:
      scores_per_block: list of length L
        scores_per_block[l]: [B, H_l] per-sample per-channel scores (abs-valued).
    """
    dev = model_device(model)
    model.eval().to(dev)

    with torch.enable_grad():  # ensure grad mode is on inside
        x = x.to(dev)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        # IMPORTANT: make input require grad *before* forward
        x.requires_grad_(True)

        # forward pass that keeps graph on x_list
        logits, x_list, caches = forward_stream_states(model, x)
        B, C = logits.shape

        if y is not None:
            y = y.to(dev)

        # choose class index per sample
        if class_source == "pred":
            class_idx = logits.argmax(dim=-1)  # [B]
        elif class_source == "true":
            assert y is not None, "y must be provided when class_source='true'"
            class_idx = y
        else:
            raise ValueError("class_source must be 'pred' or 'true'")

        # scalar target: sum of chosen-class logits
        s = logits[torch.arange(B, device=dev), class_idx].sum()
        # TODO: is above correct? Why sum? 
        
        # grads wrt x_1,...,x_L (these are still on the graph)
        grads = torch.autograd.grad(
            s,
            x_list[1:],        # x_1..x_L
            retain_graph=False,
            create_graph=False,
        )  # list length L, each [B,D]

        scores_per_block = []
        for l, blk in enumerate(model.blocks):
            g_next = grads[l]                # [B,D]
            Dmat = blk.D.weight              # [D,H]
            h = caches[l]["h"]               # [B,H]

            # alpha[b,k] = g_next[b,:] · D[:,k]
            alpha = torch.einsum("bd,dh->bh", g_next, Dmat)  # [B,H]
            S = (alpha * h).abs()            # [B,H]
            scores_per_block.append(S)

    return scores_per_block


# =========================
#  Channel sparsity losses (batch-level)
# =========================

def global_hoyer_channel_sparsity(scores_per_block, lambda_reg=1e-3, eps=1e-8):
    """
    3A: per-block normalize scores, concat, apply ONE global Hoyer.

    scores_per_block: list of [H_l] tensors (batch-averaged scores).
    """
    normed = []
    for S in scores_per_block:
        if S.numel() == 0:
            continue
        S_abs = S.abs()
        Z = S_abs.sum() + eps
        normed.append(S_abs / Z)  # sum=1 per block

    if not normed:
        return torch.tensor(0.0, device=scores_per_block[0].device)

    v = torch.cat(normed, dim=0)  # [P]
    P = v.numel()

    l1 = v.abs().sum()
    l2 = torch.sqrt((v**2).sum() + eps)
    sqrtP = torch.sqrt(torch.tensor(float(P), device=v.device))

    # Hoyer in [0,1]: 0 = dense, ~1 = 1-sparse
    hoyer = (sqrtP - l1 / (l2 + eps)) / (sqrtP - 1.0 + eps)

    # loss small when Hoyer is large (i.e. sparse)
    loss = 1.0 - hoyer
    return lambda_reg * loss


def global_entropy_channel_sparsity(scores_per_block, lambda_reg=1e-3, eps=1e-8):
    """
    3A: per-block normalize scores, concat, apply ONE global entropy penalty.

    scores_per_block: list of [H_l] tensors (batch-averaged scores).
    """
    normed = []
    for S in scores_per_block:
        if S.numel() == 0:
            continue
        S_abs = S.abs()
        Z = S_abs.sum() + eps
        normed.append(S_abs / Z)

    if not normed:
        return torch.tensor(0.0, device=scores_per_block[0].device)

    p = torch.cat(normed, dim=0)  # [P]
    p = p / (p.sum() + eps)       # global normalize

    ent = -(p * (p + eps).log()).sum()
    return lambda_reg * ent


# =========================
#  Optional: per-sample sparsity (experimental)
# =========================

def per_sample_channel_scores_from_states(
    model,
    x_list,
    caches,
    logits,
    y,
    class_source="true",
):
    """
    Per-sample, per-channel scores via grad-folded attribution.
    Returns:
      scores_per_block: list of [B,H_l]
    """
    dev = logits.device
    B, _ = logits.shape

    if class_source == "pred":
        class_idx = logits.argmax(dim=-1)
    elif class_source == "true":
        class_idx = y
    else:
        raise ValueError("class_source must be 'pred' or 'true'")

    s = logits[torch.arange(B, device=dev), class_idx].sum()

    grads = torch.autograd.grad(
        s,
        x_list[1:],
        retain_graph=True,
        create_graph=True,
    )  # list length L, each [B,D]

    scores_per_block = []
    for l, blk in enumerate(model.blocks):
        g_next = grads[l]           # [B,D]
        Dmat  = blk.D.weight        # [D,H]
        h     = caches[l]["h"]      # [B,H]

        alpha = torch.einsum("bd,dh->bh", g_next, Dmat)  # [B,H]
        S = (alpha * h).abs()       # [B,H]
        scores_per_block.append(S)

    return scores_per_block


def per_sample_global_hoyer_sparsity(scores_per_block, lambda_reg=1e-3, eps=1e-8):
    """
    Per-sample global Hoyer (3A), averaged over batch.
    scores_per_block: list of [B, H_l]
    """
    if not scores_per_block:
        raise ValueError("scores_per_block is empty")

    device = scores_per_block[0].device

    # Normalize each block per sample, then concatenate
    normalized_parts = []
    for S in scores_per_block:
        s_abs = S.abs()  # [B, H_l]
        Z = s_abs.sum(dim=1, keepdim=True) + eps  # [B, 1]
        normalized_parts.append(s_abs / Z)  # [B, H_l]

    v = torch.cat(normalized_parts, dim=1)  # [B, P]
    P = v.shape[1]

    # Compute Hoyer sparsity per sample
    l1 = v.sum(dim=1)  # [B]
    l2 = torch.sqrt((v ** 2).sum(dim=1) + eps)  # [B]
    sqrtP = torch.sqrt(torch.tensor(float(P), device=device))

    hoyer = (sqrtP - l1 / (l2 + eps)) / (sqrtP - 1.0 + eps)  # [B]

    return lambda_reg * (1.0 - hoyer).mean()


def per_sample_global_entropy_sparsity(scores_per_block, lambda_reg=1e-3, eps=1e-8):
    """
    Per-sample global entropy (3A), averaged over batch.
    scores_per_block: list of [B, H_l]
    """
    if not scores_per_block:
        raise ValueError("scores_per_block is empty")

    # Normalize each block per sample, then concatenate
    normalized_parts = []
    for S in scores_per_block:
        s_abs = S.abs()  # [B, H_l]
        Z = s_abs.sum(dim=1, keepdim=True) + eps  # [B, 1]
        normalized_parts.append(s_abs / Z)  # [B, H_l]

    p = torch.cat(normalized_parts, dim=1)  # [B, P]

    # Normalize again to get proper probability distribution per sample
    p = p / (p.sum(dim=1, keepdim=True) + eps)  # [B, P]

    # Compute entropy per sample
    ent = -(p * (p + eps).log()).sum(dim=1)  # [B]

    return lambda_reg * ent.mean()


# =========================
#  MNIST data, training, eval
# =========================

def make_mnist_loaders(batch_size=128, kind="mnist"):
    from torchvision import datasets, transforms

    if kind == "mnist":
        mean, std = (0.1307,), (0.3081,)
        DS = datasets.MNIST
    elif kind == "fashion":
        # standard FashionMNIST stats
        mean, std = (0.2860,), (0.3530,)
        DS = datasets.FashionMNIST
    else:
        raise ValueError(f"unknown kind={kind!r}, use 'mnist' or 'fashion'")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train = DS(root="./data", train=True, download=True, transform=tfm)
    test  = DS(root="./data", train=False, download=True, transform=tfm)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def train_mnist(model, train_loader, epochs=1, lr=1e-3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model.forward(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"epoch {ep+1}/{epochs}  loss={loss.item():.4f}")

    return model


@torch.no_grad()
def eval_acc(model, loader, device=None):
    if device is None:
        device = model_device(model)
    model.eval().to(device)
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model.forward(x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_mnist_with_channel_sparsity(
    model,
    train_loader,
    epochs=1,
    lr=1e-3,
    lambda_reg=1e-3,
    blocks_to_reg=None,
    device=None,
    sparsity_loss_type="hoyer",  # "hoyer" or "entropy"
    weight_decay=0.0,
):
    """
    Main training path: batch-averaged channel sparsity (not per-sample).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits, x_list, caches = forward_stream_states(model, x)

            # require_grad on stream states for channel scores
            for t in x_list:
                t.requires_grad_(True)

            ce = F.cross_entropy(logits, y)

            scores = channel_scores_grad_folded_from_states(
                model, x_list, caches, logits, y, blocks_to_reg=blocks_to_reg
            )

            if sparsity_loss_type == "hoyer":
                sparsity = global_hoyer_channel_sparsity(scores, lambda_reg=lambda_reg)
            elif sparsity_loss_type == "entropy":
                sparsity = global_entropy_channel_sparsity(scores, lambda_reg=lambda_reg)
            else:
                raise ValueError(f"Invalid sparsity_loss_type: {sparsity_loss_type}")

            loss = ce + sparsity

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"epoch {ep+1}/{epochs}  ce={ce.item():.4f}  sparsity={sparsity.item():.4f}")

    return model


def train_mnist_with_per_sample_sparsity(
    model,
    train_loader,
    epochs=1,
    lr=1e-3,
    lambda_reg=1e-3,
    device=None,
    sparsity_kind="hoyer",  # "hoyer" or "entropy"
    class_source="true",
    weight_decay=0.0,
):
    """
    Experimental: per-sample sparsity; slower and empirically worse.
    Not used in the main path, kept for reference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits, x_list, caches = forward_stream_states(model, x)

            for t in x_list:
                t.requires_grad_(True)

            logits2 = model.head(x_list[-1])
            ce = F.cross_entropy(logits2, y)

            scores_per_block = per_sample_channel_scores_from_states(
                model, x_list, caches, logits2, y, class_source=class_source
            )

            if sparsity_kind == "hoyer":
                sparsity = per_sample_global_hoyer_sparsity(scores_per_block, lambda_reg=lambda_reg)
            elif sparsity_kind == "entropy":
                sparsity = per_sample_global_entropy_sparsity(scores_per_block, lambda_reg=lambda_reg)
            else:
                raise ValueError("sparsity_kind must be 'hoyer' or 'entropy'")

            loss = ce + sparsity

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"[per-sample] epoch {ep+1}/{epochs}  ce={ce.item():.4f}  sparsity={sparsity.item():.4f}")

    return model


# =========================
#  Channel masks + SVD
# =========================
def channel_attrib_batch_grad_folded(model, x, y=None, class_source="pred"):
    """
    Per-sample grad-folded channel attributions.

    Returns:
      scores_per_block: list of length L
        scores_per_block[l]: [B, H_l] per-sample per-channel scores (abs-valued).
    """
    dev = model_device(model)
    model.eval().to(dev)

    with torch.enable_grad():  # ensure grad mode is on inside
        x = x.to(dev)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        # IMPORTANT: make input require grad *before* forward
        x.requires_grad_(True)

        # forward pass that keeps graph on x_list
        logits, x_list, caches = forward_stream_states(model, x)
        B, C = logits.shape

        if y is not None:
            y = y.to(dev)

        # choose class index per sample
        if class_source == "pred":
            class_idx = logits.argmax(dim=-1)  # [B]
        elif class_source == "true":
            assert y is not None, "y must be provided when class_source='true'"
            class_idx = y
        else:
            raise ValueError("class_source must be 'pred' or 'true'")

        # scalar target: sum of chosen-class logits
        s = logits[torch.arange(B, device=dev), class_idx].sum()

        # grads wrt x_1,...,x_L (these are still on the graph)
        grads = torch.autograd.grad(
            s,
            x_list[1:],        # x_1..x_L
            retain_graph=False,
            create_graph=False,
        )  # list length L, each [B,D]

        scores_per_block = []
        for l, blk in enumerate(model.blocks):
            g_next = grads[l]                # [B,D]
            Dmat = blk.D.weight              # [D,H]
            h = caches[l]["h"]               # [B,H]

            # alpha[b,k] = g_next[b,:] · D[:,k]
            alpha = torch.einsum("bd,dh->bh", g_next, Dmat)  # [B,H]
            S = (alpha * h).abs()            # [B,H]
            scores_per_block.append(S)

    return scores_per_block



def flatten_channel_attrib(scores_per_block):
    """
    scores_per_block: list of [B,H_l]
    Returns:
      A: [B,P]
    """
    return torch.cat(scores_per_block, dim=1)


def channel_mask_from_flat(a, keep_frac=0.99):
    """
    a: [P] scores
    Returns mask [P] bool keeping keep_frac of |a| mass.
    """
    abs_a = a.abs()
    vals, idx = torch.sort(abs_a, descending=True)
    csum = torch.cumsum(vals, 0)
    total = vals.sum() + 1e-12
    cutoff = keep_frac * total

    keep_n = int((csum <= cutoff).sum().item())
    keep_n = max(1, keep_n)

    keep_idx = idx[:keep_n]
    m = torch.zeros_like(a, dtype=torch.bool)
    m[keep_idx] = True
    return m


@torch.no_grad()
def collect_channel_mask_matrix(
    model,
    loader,
    keep_frac=0.99,
    max_samples=None,
    class_source="pred",
    device=None,
):
    """
    Build per-input channel masks (per channel, not per-weight).

    Returns:
      M: [N,P] float32 0/1 matrix of masks.
    """
    if device is None:
        device = model_device(model)
    model.eval().to(device)

    masks_list = []
    seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        scores_per_block = channel_attrib_batch_grad_folded(
            model, x, y=y, class_source=class_source
        )
        A = flatten_channel_attrib(scores_per_block)  # [B,P]
        B, P = A.shape

        for i in range(B):
            m = channel_mask_from_flat(A[i], keep_frac=keep_frac)
            masks_list.append(m.to(torch.float32).cpu())
            seen += 1
            if max_samples is not None and seen >= max_samples:
                break

        if max_samples is not None and seen >= max_samples:
            break

    if not masks_list:
        return torch.empty(0, 0)

    M = torch.stack(masks_list, dim=0)  # [N,P]
    return M


def channel_scores_grad_folded_from_states(
    model,
    x_list,
    caches,
    logits,
    y,
    blocks_to_reg=None,
):
    """
    Batch-averaged gradient-folded channel scores.

    x_list: [x0, x1, ..., xL] (from forward_stream_states)
    caches: per-block caches with 'h'
    logits: [B,C]
    y: [B] true labels

    Returns:
      scores_per_block: list [S_0,...,S_{L-1}] where S_l is [H_l]
        S_l[k] = E_b |alpha_{b,k} * h_{b,k}| (avg over batch).
    """
    dev = logits.device
    B = logits.size(0)

    # scalar target: sum of correct-class logits
    s = logits[torch.arange(B, device=dev), y].sum()

    # grads wrt x_1,...,x_L
    grads = torch.autograd.grad(
        s,
        x_list[1:],      # x_1..x_L
        retain_graph=True,
        create_graph=True,   # so sparsity loss is differentiable wrt params
    )  # list length L, each [B,D]

    scores_per_block = []

    if blocks_to_reg is None:
        blocks_to_reg = range(len(model.blocks))

    for l, blk in enumerate(model.blocks):
        H, _ = blk.L.weight.shape

        if l not in blocks_to_reg:
            scores_per_block.append(torch.zeros(H, device=dev, dtype=logits.dtype))
            continue

        g_next = grads[l]              # [B,D]
        Dmat = blk.D.weight            # [D,H]
        h = caches[l]["h"]             # [B,H]

        # alpha[b,k] = g_next[b,:] · D[:,k]
        alpha = torch.einsum("bd,dh->bh", g_next, Dmat)  # [B,H]

        # per-channel importance: E_b |alpha * h|
        I = (alpha * h).abs().mean(dim=0)  # [H]
        scores_per_block.append(I)

    return scores_per_block


def svd_masks(M, rank=16, center=True):
    """
    SVD on mask matrix M [N,P]:
      M ≈ U S Vh
    """
    if M.numel() == 0:
        return None
    X = M
    if center:
        X = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    r = min(rank, S.numel())
    return U[:, :r], S[:r], Vh[:r, :]


def build_channel_index_map(model):
    """
    Map flat channel index -> (block_idx, channel_idx).
    """
    idx_map = []
    for bi, blk in enumerate(model.blocks):
        H, _ = blk.L.weight.shape
        for k in range(H):
            idx_map.append((bi, k))
    return idx_map


def top_channels_in_component(model, V_row, top_k=20):
    """
    V_row: [P] (one SVD right-singular vector).
    Returns list of (|v|, block_idx, channel_idx).
    """
    idx_map = build_channel_index_map(model)
    vals = V_row.abs()
    top = torch.topk(vals, k=min(top_k, vals.numel()))
    out = []
    for score, flat_i in zip(top.values.tolist(), top.indices.tolist()):
        bi, k = idx_map[flat_i]
        out.append((score, bi, k))
    return out


# =========================
#  Diagnostics: Hoyer / entropy on vectors
# =========================

def hoyer_of_vector(v, eps=1e-8):
    v_abs = v.abs()
    l1 = v_abs.sum()
    l2 = torch.sqrt((v_abs**2).sum() + eps)
    H = v.numel()
    # "Hoyer-like" score in [0,1]; 0 = dense-ish, ~1 = sparse-ish
    return l1 / (torch.sqrt(torch.tensor(float(H), device=v.device)) * l2 + eps)


def entropy_of_vector(v, eps=1e-8):
    v_abs = v.abs()
    p = v_abs / (v_abs.sum() + eps)
    ent = -(p * (p + eps).log()).sum()
    return ent

def collect_per_sample_channel_sparsity(
    model,
    loader,
    device=None,
    class_source="pred",   # "pred" or "true"
    max_samples=None,
    eps=1e-8,
):
    """
    For each datapoint:
      - get channel scores via grad-folded attribution
      - flatten across blocks -> [P]
      - normalize to a distribution over channels
      - compute Hoyer + entropy of that distribution

    Returns:
      hoyers  : 1D tensor [N]
      entropies : 1D tensor [N]
    """
    if device is None:
        device = model_device(model)
    model.eval().to(device)

    all_hoyer = []
    all_ent = []
    seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # need grad for attribution
        scores_per_block = channel_attrib_batch_grad_folded(
            model, x, y=y, class_source=class_source
        )  # list of [B, H_l]

        A = flatten_channel_attrib(scores_per_block)  # [B, P]
        B, P = A.shape

        for i in range(B):
            v = A[i]          # [P]
            v_abs = v.abs()
            v_norm = v_abs / (v_abs.sum() + eps)

            h = hoyer_of_vector(v_norm)
            e = entropy_of_vector(v_norm)

            all_hoyer.append(h.detach().cpu())
            all_ent.append(e.detach().cpu())

            seen += 1
            if max_samples is not None and seen >= max_samples:
                break

        if max_samples is not None and seen >= max_samples:
            break

    if not all_hoyer:
        return torch.empty(0), torch.empty(0)

    hoyers = torch.stack(all_hoyer)   # [N]
    ents   = torch.stack(all_ent)     # [N]
    return hoyers, ents

# =========================
#  High-level demos
# =========================
@torch.no_grad()
def channel_argmax_hist(model, loader, device, class_source="pred"):
    model.eval().to(device)
    counts = None

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        scores_per_block = channel_attrib_batch_grad_folded(
            model, x, y=y, class_source=class_source
        )  # list of [B,H_l]
        A = flatten_channel_attrib(scores_per_block)  # [B,P]

        idx = A.argmax(dim=1)  # [B] most-used channel per sample
        B, P = A.shape
        if counts is None:
            counts = torch.zeros(P, dtype=torch.long)
        for i in idx:
            counts[i] += 1

    return counts

def channel_mask_svd_demo():
    """
    Main demo used in __main__:
      - train bilinear ResNet with channel sparsity
      - collect per-input channel masks
      - SVD on mask matrix
      - inspect first SVD component & Hoyer/entropy.
    """
    train_loader, test_loader = make_mnist_loaders(batch_size=256, kind="fashion")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = BilinearResNetMNIST(
    #     in_dim=784, hidden_dim=4, n_blocks=2, out_dim=10, bias=False
    # ).to(device)
    model = BilinearResNetMNIST(
        in_dim=784, hidden_dim=6, n_blocks=4, out_dim=10, bias=False
    ).to(device)

    per_sample_sparsity = True
    loss_type = "hoyer"
    # loss_type = "entropy"
    lambda_reg = 1
    # Train with batch-averaged channel sparsity
    if per_sample_sparsity:
        model = train_mnist_with_per_sample_sparsity(
            model,
            train_loader,
            epochs=10,
            lr=1e-3,
            lambda_reg=lambda_reg,         # strong sparsity
            # blocks_to_reg=[0, 1],
            device=device,
            sparsity_kind=loss_type,  # or "entropy"
            weight_decay=1e-3,
        )
    else:
        model = train_mnist_with_channel_sparsity(
            model,
            train_loader,
            epochs=10,
            lr=1e-3,
            lambda_reg=lambda_reg,         # strong sparsity
            blocks_to_reg=[0, 1],
            device=device,
            sparsity_loss_type=loss_type,  # or "entropy"
            weight_decay=1e-3,
        )
        
    acc = eval_acc(model, test_loader, device=device)
    print("test acc:", acc)

    # collect channel masks
    M = collect_channel_mask_matrix(
        model,
        test_loader,
        keep_frac=0.99,
        max_samples=1000,
        class_source="pred",
        device=device,
    )
    print("mask matrix shape:", tuple(M.shape))

    fac = svd_masks(M, rank=8, center=True)
    if fac is None:
        print("No masks collected.")
        return

    U, S, Vh = fac
    print("top singular values:", S.tolist())

    # inspect first component
    comp0 = top_channels_in_component(model, Vh[0], top_k=10)
    print("\nComponent 0 top channels (|v|, block, channel):")
    for score, bi, k in comp0:
        print(f"  |v|={score:.4f}  block={bi}  channel={k}")

    return model, M, fac


# =========================
#  __main__
# =========================

from matplotlib import pyplot as plt

def plot_sparsity_hist(hoyers, ents, bins=50):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(hoyers.numpy(), bins=bins)
    plt.title("Per-sample Hoyer (channels)")
    plt.xlabel("Hoyer")
    plt.ylabel("#samples")

    plt.subplot(1,2,2)
    plt.hist(ents.numpy(), bins=bins)
    plt.title("Per-sample entropy (channels)")
    plt.xlabel("Entropy")
    plt.ylabel("#samples")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model, M, fac = channel_mask_svd_demo()
    U, S, Vh = fac

    print("\nPer-sample U rows (first 10 samples):")
    for i in range(min(10, U.shape[0])):
        u = U[i]
        hoyer = hoyer_of_vector(u)
        entropy = entropy_of_vector(u)
        print(f"sample {i:2d}: hoyer={hoyer:.4f}  entropy={entropy:.4f}")
        print("  u = [" + ", ".join(f"{u[j].item():.3f}" for j in range(len(u))) + "]")

    print("\nRight singular vectors V (components over channels):")
    for comp_idx in range(Vh.shape[0]):
        v = Vh[comp_idx]
        hoyer = hoyer_of_vector(v)
        entropy = entropy_of_vector(v)
        print(f"comp {comp_idx:2d}: hoyer={hoyer:.4f}  entropy={entropy:.4f}")
        print("  v = [" + ", ".join(f"{v[j].item():.3f}" for j in range(len(v))) + "]")
    
    avg_hoyer = 0
    N = 1000
    for i in range(N):
        u = U[i]
        avg_hoyer += hoyer_of_vector(u)
    avg_hoyer /= N
    print(f"avg_hoyer={avg_hoyer:.4f}")
        # inspect first component
    for comp_idx in range(Vh.shape[0]):
        comp = top_channels_in_component(model, Vh[comp_idx], top_k=10)
        print(f"\nComponent {comp_idx} top channels (|v|, block, channel):")
        for score, bi, k in comp:
            print(f"  |v|={score:.4f}  block={bi}  channel={k}")

    train_loader, test_loader = make_mnist_loaders(batch_size=256, kind="fashion")
    device = model_device(model)
    counts = channel_argmax_hist(model, test_loader, device, class_source="pred")
    from matplotlib import pyplot as plt
    plt.bar(range(counts.shape[0]), counts.tolist())
    plt.show()


    hoyers, ents = collect_per_sample_channel_sparsity(
        model,
        test_loader,
        device=device,
        class_source="pred",
        max_samples=2000,
    )

    print("per-sample Hoyer:   mean={:.3f}  std={:.3f}".format(hoyers.mean().item(), hoyers.std().item()))
    print("per-sample entropy: mean={:.3f}  std={:.3f}".format(ents.mean().item(), ents.std().item()))

    plot_sparsity_hist(hoyers, ents)

    all_scores = []
    all_labels = []
    all_preds = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        # need grad for attribution
        scores_per_block = channel_attrib_batch_grad_folded(
            model, x, y=y, class_source="true"
        )  # list of [B, H_l]
        all_scores.append(scores_per_block)
        all_labels.append(y)
        # all_preds.append(scores_per_block.argmax(dim=-1))

        A = flatten_channel_attrib(scores_per_block)  # [B, P]
        for i in range(10):
            print(A[i].topk(5))
        break
    #%%
    # visualize the first 10 samples
#     for i in range(10):
#         plt.imshow(x[i].reshape(28, 28), cmap="gray")
#         plt.show()

#     N = 0
#     u = model.blocks[N].L.weight
#     v = model.blocks[N].R.weight
#     d = model.blocks[N].D.weight

#     C = 0
#     u1 = u[C]
#     v1 = v[C]
#     d1 = d[C]
#     print(u1.shape, v1.shape, d1.shape)

#     intM = u1[:, None] @ v1[None, :]
#     plt.imshow(intM.detach().cpu(), cmap="viridis")
#     plt.colorbar()
#     plt.show()
# def visualize_interaction_matrix(u, v):
#     intM = u[:, None] @ v[None, :]
#     plt.imshow(intM.detach().cpu(), cmap="viridis")
#     plt.colorbar()
#     plt.show()

#%% 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, '/home/loganriggs/Coding/toy_models_of_tensor_networks')
from binary_sae import SparseORAE
device = "cuda" if torch.cuda.is_available() else "cpu"
sae = SparseORAE(
    input_dim=M.shape[1],
    latent_dim=30,
    top_k=2,
    min_threshold=0.5,
).to(device)
sae.fit(M.to(device), epochs=10_000, batch_size=1024, lr=1e-3, print_every=100)

#%%
# Demo: explain a single sample
print("\n" + "="*60)
print("EXAMPLE: Explaining sample 0")
print("="*60)

result = sae.explain(M[1].to(device))

print(f"\nInput:   {M[1].numpy().astype(int)}")
print(f"Recon:   {result['reconstruction'].cpu().numpy().astype(int)}")
print(f"Accuracy: {result['accuracy']:.2%}")
print(f"\nActive features: {result['active_features']}")

for idx, activation in result['active_features']:
    pattern = result['patterns'][idx].cpu().numpy().astype(int)
    print(f"  Feature {idx} (act={activation:.3f}): {pattern}")

# Demo: batch query
print("\n" + "="*60)
print("EXAMPLE: Active features for samples 0-4")
print("="*60)


features = sae.get_active_features(M[0:20].to(device))
for i, (f, y_class) in enumerate(zip(features, y)):
    print(f"Sample {i}| Class {y_class}: {f}")
# %%
# Generalize
num_layers = len(model.blocks)

for layer_idx in range(num_layers):
    num_plots = num_layers - layer_idx - 1
    if num_plots == 0:
        continue
    
    # Precompute all data and global min/max for this layer
    data_list = []
    global_max = float('-inf')
    global_min = float('inf')
    
    d = model.blocks[layer_idx].D.weight
    
    for next_layer_idx in range(layer_idx + 1, num_layers):
        u2 = model.blocks[next_layer_idx].L.weight
        v2 = model.blocks[next_layer_idx].R.weight
        int1 = ((d.T @ u2.T) * (d.T @ v2.T)).T
        data = int1.detach().cpu()
        data_list.append((next_layer_idx, data, u2.shape[0]))
        
        global_max = max(global_max, data.max().item())
        global_min = min(global_min, data.min().item())
    
    # Make symmetric around zero
    global_abs_max = max(abs(global_max), abs(global_min))
    vmin, vmax = -global_abs_max, global_abs_max
    
    # Create subplots
    ncols = min(num_plots, 4)  # Max 4 columns
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f"Down_proj at layer {layer_idx} composed with U & V at subsequent layers", fontsize=14)
    
    for plot_idx, (next_layer_idx, data, u2_dim) in enumerate(data_list):
        row, col = plot_idx // ncols, plot_idx % ncols
        ax = axes[row, col]
        
        im = ax.imshow(data, cmap="RdBu", vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f"Layer {next_layer_idx}")

        ax.set_xlabel("Input D_proj")
        ax.set_ylabel("U, V hidden channels")
        
        ax.set_xticks(range(d.shape[1]))
        ax.set_xticklabels([f"d{i}" for i in range(d.shape[1])])
        ax.set_yticks(range(u2_dim))
        ax.set_yticklabels([f"h{i}" for i in range(u2_dim)])
        
        # # Grid on cell boundaries

        ax.set_xticks([x - 0.5 for x in range(1, d.shape[1])], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, u2_dim)], minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Hide unused subplots
    for plot_idx in range(num_plots, nrows * ncols):
        row, col = plot_idx // ncols, plot_idx % ncols
        axes[row, col].axis('off')

    
    # Single colorbar for the whole figure
    fig.colorbar(im, ax=axes, shrink=0.6, label="Interaction strength")
    # plt.tight_layout()
    plt.show()

#%%
# Head interactions with each layer's D_proj
num_layers = len(model.blocks)
head = model.head.weight.detach().cpu()

# FashionMNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for layer_idx in range(num_layers):
    d = model.blocks[layer_idx].D.weight.detach().cpu()
    data = head @ d
    
    # Local symmetric scaling
    local_abs_max = max(abs(data.max().item()), abs(data.min().item()))
    vmin, vmax = -local_abs_max, local_abs_max
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Head composed with Down_proj at layer {layer_idx}")
    ax.set_xlabel("D_proj channels")
    ax.set_ylabel("Output class")
    
    im = ax.imshow(data, cmap="RdBu", vmin=vmin, vmax=vmax, aspect='auto')
    
    # X ticks (D_proj dimensions)
    d_dim = d.shape[0]
    if d_dim <= 20:
        ax.set_xticks(range(d_dim))
        ax.set_xticklabels([f"d{i}" for i in range(d_dim)], rotation=45, ha='right')
    else:
        ax.set_xlabel(f"D_proj channels (dim={d_dim})")
    
    # Y ticks (class names)
    ax.set_yticks(range(10))
    ax.set_yticklabels(class_names)
    
    # Grid on cell boundaries
    if d_dim <= 50:
        ax.set_xticks([x - 0.5 for x in range(1, d_dim)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 10)], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.colorbar(im, label="Interaction strength")
    plt.tight_layout()
    plt.show()

#%%
# Head interactions with each layer's D_proj
num_layers = len(model.blocks)
head = model.head.weight.detach().cpu()

# FashionMNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6))
fig.suptitle("Head composed with Down_proj at each layer", fontsize=14)

if num_layers == 1:
    axes = [axes]

for layer_idx in range(num_layers):
    ax = axes[layer_idx]
    d = model.blocks[layer_idx].D.weight.detach().cpu()
    data = head @ d
    
    # Local symmetric scaling
    local_abs_max = max(abs(data.max().item()), abs(data.min().item()))
    vmin, vmax = -local_abs_max, local_abs_max
    
    im = ax.imshow(data, cmap="RdBu", vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f"Layer {layer_idx}")
    ax.set_xlabel("D_proj channels")
    
    # X ticks (D_proj dimensions)
    d_dim = d.shape[0]
    if d_dim <= 20:
        ax.set_xticks(range(d_dim))
        ax.set_xticklabels([f"d{i}" for i in range(d_dim)], rotation=45, ha='right')
    else:
        ax.set_xlabel(f"D_proj channels (dim={d_dim})")
    
    # Y ticks - only show labels on leftmost plot
    ax.set_yticks(range(10))
    if layer_idx == 0:
        ax.set_yticklabels(class_names)
        ax.set_ylabel("Output class")
    else:
        ax.set_yticklabels([])
    
    # Grid on cell boundaries
    if d_dim <= 50:
        ax.set_xticks([x - 0.5 for x in range(1, d_dim)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 10)], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Individual colorbar for each subplot
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()

#%%
# Explain a single sample, these are for the correct class
# We should see a match up w/ the relevant heads for a class & the sparse channels here
import numpy as np
d_idx = 4
result = sae.explain(M[d_idx].to(device))
print(f"Class {y[d_idx].item()}")
for idx, activation in result['active_features']:
    pattern = result['patterns'][idx].cpu().numpy().astype(int)
    print(f"  Feature {idx} (act={activation:.3f}): {pattern}")
    nz = pattern.nonzero()[0]
    row, col = np.floor(nz/6), nz%6
    print(row, col)
# %%    
target_input = x[d_idx].flatten()

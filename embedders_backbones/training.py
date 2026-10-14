"""Shared self-supervised training loop (CPU-first) for every NEMBA backbone.

Objectives
----------
``masked``       Masked-span reconstruction (BERT / Zerveas style). The masked
                 frames are zeroed in the *input* before any layer sees them, so
                 no architecture can copy the answer through its receptive
                 field (the v0.1 TCN family and EmbTCN could).
``contrastive``  Time-contrastive InfoNCE (CEBRA-time style): two frames a few
                 bins apart in the same window are a positive pair, every other
                 frame in the batch is a negative.
``hybrid``       ``masked + contrastive_weight * contrastive``.

Speed (why it is fast without a GPU)
------------------------------------
* a fixed optimisation *step budget* instead of epochs, so cost does not grow
  with recording length;
* random window crops (more diverse batches per step than fixed tiles);
* early stopping on a held-out contiguous block of the recording;
* warm-up + cosine learning-rate schedule (converges in fewer steps);
* batched, overlap-add inference with tapered windows (no seams);
* ``speed`` presets that also shrink model width for the ``fast`` mode.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import backbones as BK
from ._embtcn import make_span_mask, masked_reconstruction_loss

# --------------------------------------------------------------------------- #
# user-facing presets                                                          #
# --------------------------------------------------------------------------- #
# max_steps : optimisation steps budget (early stopping may stop sooner)
# win_len   : frames per training window
# batch     : windows per optimisation step
# eval_every: steps between two validation checks
# patience  : validation checks without improvement before stopping
# width     : multiplier on the backbone's default hidden width (d_model)
# time_budget_s : wall-clock cap on training (None = no cap). Caps make CPU run
#             time predictable but results machine-dependent; use
#             ``time_budget_s=None`` (or 'accurate') for exact reproducibility.
SPEED_PRESETS = {
    "fast":     dict(max_steps=300,  win_len=128, batch=32, eval_every=25,  patience=4,
                     width=0.5, time_budget_s=90.0),
    "balanced": dict(max_steps=1000, win_len=256, batch=32, eval_every=50,  patience=5,
                     width=1.0, time_budget_s=600.0),
    "accurate": dict(max_steps=4000, win_len=256, batch=32, eval_every=100, patience=8,
                     width=1.0, time_budget_s=None),
}
OBJECTIVES = ("masked", "contrastive", "hybrid")

# Channel-token models (one token per channel x frame) get their batch capped so
# that one step never processes more than this many tokens (CPU memory / time).
TOKENS_PER_STEP = 2 ** 16


def resolve_device(device="auto"):
    """'auto' -> CUDA when available, else CPU."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def configure_threads(n_threads=None):
    """Optionally pin the number of CPU threads; flush denormals (free speed-up
    on x86 when weights become tiny)."""
    if n_threads:
        torch.set_num_threads(int(n_threads))
    try:
        torch.set_flush_denormal(True)
    except Exception:  # noqa: BLE001
        pass


def resolve_settings(speed="fast", **user):
    """Merge a speed preset with explicit user values (``None`` = use preset)."""
    if speed not in SPEED_PRESETS:
        raise ValueError(f"speed must be one of {list(SPEED_PRESETS)}, got {speed!r}")
    out = dict(SPEED_PRESETS[speed])
    out.update({k: v for k, v in user.items() if v is not None})
    return out


# --------------------------------------------------------------------------- #
# objectives                                                                   #
# --------------------------------------------------------------------------- #

def apply_input_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the masked frames of x [B, C, T] (mask [B, T], True = hidden).

    Inputs are z-scored, so 0 is the per-channel mean: the model sees "no
    information" at masked frames, exactly what masked modelling requires."""
    return x.masked_fill(mask[:, None, :], 0.0)


def time_contrastive_loss(z: torch.Tensor, *, n_pairs: int = 8, max_offset: int = 10,
                          temperature: float = 0.2, generator=None) -> torch.Tensor:
    """Symmetric InfoNCE over time-offset positive pairs.

    z : [B, E, L] per-frame embeddings of B windows.
    For every window, ``n_pairs`` anchors t are drawn with positives t + d,
    d ~ U{1..max_offset}. Similarities are cosine / temperature; all other
    positives in the batch act as negatives (B * n_pairs - 1 per anchor).
    """
    B, E, L = z.shape
    if L < 2:
        return z.sum() * 0.0
    dev = z.device
    max_offset = int(max(1, min(max_offset, L - 1)))
    off = torch.randint(1, max_offset + 1, (B, n_pairs), generator=generator)
    t0 = (torch.rand(B, n_pairs, generator=generator) * (L - off)).long()
    t1 = t0 + off
    b = torch.arange(B)[:, None].expand(B, n_pairs)
    zt = z.transpose(1, 2)                                          # [B, L, E]
    a = F.normalize(zt[b.to(dev), t0.to(dev)].reshape(-1, E), dim=-1)
    p = F.normalize(zt[b.to(dev), t1.to(dev)].reshape(-1, E), dim=-1)
    logits = a @ p.T / float(temperature)
    labels = torch.arange(a.shape[0], device=dev)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _objective_loss(net, xb, mask, cfg, generator):
    """One forward pass + the requested loss (plus any model auxiliary loss)."""
    obj = cfg["objective"]
    if obj == "contrastive":
        out = net(xb, None)
        loss = time_contrastive_loss(out.embeddings, n_pairs=cfg["n_pairs"],
                                     max_offset=cfg["contrastive_offset"],
                                     temperature=cfg["temperature"], generator=generator)
    else:
        out = net(apply_input_mask(xb, mask), mask)
        loss = masked_reconstruction_loss(xb, out.reconstruction, mask)
        if obj == "hybrid":
            loss = loss + cfg["contrastive_weight"] * time_contrastive_loss(
                out.embeddings, n_pairs=cfg["n_pairs"], max_offset=cfg["contrastive_offset"],
                temperature=cfg["temperature"], generator=generator)
    aux = getattr(out, "aux_loss", None)
    if aux is not None:
        loss = loss + aux
    return loss


# --------------------------------------------------------------------------- #
# data sampling                                                                #
# --------------------------------------------------------------------------- #

def make_windows(X, L, stride):
    """Fixed tiling of X [C, T] into windows [W, C, L] (kept for compatibility)."""
    C, T = X.shape
    if T <= L:
        pad = np.zeros((C, L), dtype=X.dtype); pad[:, :T] = X
        return pad[None]
    starts = list(range(0, T - L + 1, stride))
    if starts[-1] != T - L:
        starts.append(T - L)
    return np.stack([X[:, s:s + L] for s in starts], axis=0)


def _n_windows(T, L, stride):
    """Number of windows ``make_windows`` would produce (legacy epoch size)."""
    if T <= L:
        return 1
    n = len(range(0, T - L + 1, stride))
    return n + (0 if (n - 1) * stride == T - L else 1)


def split_train_val(T, L, val_frac):
    """Contiguous hold-out: the last ``val_frac`` of the recording validates.

    Returns ((train_start, train_stop), (val_start, val_stop) | None). A
    contiguous block (not random windows) avoids the leakage that overlapping
    windows would create between train and validation."""
    if not val_frac or val_frac <= 0:
        return (0, T), None
    n_val = int(round(val_frac * T))
    if n_val < L or T - n_val < 2 * L:
        return (0, T), None
    return (0, T - n_val), (T - n_val, T)


def _sample_windows(Xt, seg, L, B, generator):
    """B random crops of length L from segment [s0, s1) of Xt [C, T] -> [B, C, L]."""
    s0, s1 = seg
    starts = torch.randint(s0, s1 - L + 1, (B,), generator=generator)
    idx = (starts[:, None] + torch.arange(L)).to(Xt.device)
    return Xt[:, idx].permute(1, 0, 2).contiguous()


def _val_batches(Xt, seg, L, max_windows=64):
    """Non-overlapping tiles of the validation block -> [W, C, L]."""
    s0, s1 = seg
    starts = list(range(s0, s1 - L + 1, L))[:max_windows]
    return torch.stack([Xt[:, s:s + L] for s in starts], dim=0)


# --------------------------------------------------------------------------- #
# training                                                                     #
# --------------------------------------------------------------------------- #

def train(name, X, *, embedding_dim=32, speed="fast", max_steps=None, epochs=None,
          win_len=None, batch=None, stride=None, lr=1e-3, weight_decay=1e-4,
          mask_ratio=0.4, mask_span=16, objective="masked", contrastive_weight=0.5,
          temperature=0.2, contrastive_offset=10, n_pairs=8, val_frac=0.1,
          eval_every=None, patience=None, device="auto", seed=0, log=print,
          model_kwargs=None, n_threads=None, time_budget_s="preset", progress=None):
    """Train backbone ``name`` on X [C, T] (z-scored). Returns (net, history).

    ``history`` holds the loss curves (``step``, ``train_loss``, ``val_loss``),
    ``best_step``, ``stopped_early``, ``seconds`` and the resolved settings.
    ``epochs`` (legacy) is converted to an equivalent step budget.
    ``progress(step, max_steps, train_loss, val_loss)`` is called at every
    validation check (used by the GUI progress bar).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    s = resolve_settings(speed, max_steps=max_steps, win_len=win_len, batch=batch,
                         eval_every=eval_every, patience=patience)
    if time_budget_s == "preset":                    # None = explicitly no cap
        time_budget_s = s.get("time_budget_s")
    configure_threads(n_threads)
    dev = resolve_device(device)
    torch.manual_seed(seed); np.random.seed(seed)
    X = np.ascontiguousarray(X, dtype=np.float32)
    C, T = X.shape
    L = int(min(s["win_len"], T))
    if L < 8:
        raise ValueError(f"recording too short ({T} frames); need at least 8")

    mk = dict(BK.width_overrides(name, s["width"]))
    mk.update(model_kwargs or {})
    net = BK.make_model(name, C, embedding_dim=embedding_dim, max_len=max(L, 8), **mk).to(dev)
    if BK.needs_graph(name):
        net.set_graph(BK.neuron_graph(X, top_k=min(8, max(C - 1, 1))))

    t_start = time.perf_counter()
    history = dict(step=[], train_loss=[], val_loss=[], lr=[], best_step=0,
                   best_val=float("nan"), stopped_early=False, steps_done=0,
                   seconds=0.0, objective=objective, win_len=L, batch=s["batch"],
                   speed=speed, n_params=int(sum(p.numel() for p in net.parameters())))

    # --- no-training baselines (e.g. PCA) fit in closed form --------------- #
    if not getattr(net, "trainable", True):
        net.fit_numpy(X)
        history["seconds"] = time.perf_counter() - t_start
        if log:
            log(f"  [{name}] closed-form fit in {history['seconds']:.2f}s (no SGD)")
        return net, history

    # --- step budget ------------------------------------------------------- #
    B = int(s["batch"])
    if getattr(net, "channel_tokens", False) and B * C * L > TOKENS_PER_STEP:
        B = max(2, TOKENS_PER_STEP // (C * L))
        if log:
            log(f"  [{name}] one token per channel x frame: batch capped to {B} "
                f"({C} channels x {L} frames); consider input_pca for wide inputs")
    history["batch"] = B
    steps = int(s["max_steps"])
    if epochs is not None and max_steps is None:     # legacy epochs -> steps
        steps = max(1, int(epochs) * math.ceil(_n_windows(T, L, stride or L // 2) / B))
    eval_every = max(1, int(s["eval_every"]))
    patience = max(1, int(s["patience"]))

    train_seg, val_seg = split_train_val(T, L, val_frac)
    Xt = torch.from_numpy(X).to(dev)
    gen = torch.Generator().manual_seed(seed)
    val_x = _val_batches(Xt, val_seg, L) if val_seg is not None else None

    cfg = dict(objective=objective, contrastive_weight=contrastive_weight,
               temperature=temperature, contrastive_offset=contrastive_offset, n_pairs=n_pairs)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    warm = max(1, int(0.05 * steps))

    def lr_at(step):                                  # warm-up then cosine to 10 %
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, steps - warm)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def validate():
        if val_x is None:
            return float("nan")
        net.eval()
        vgen = torch.Generator().manual_seed(seed + 12345)   # identical masks / pairs each time
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, val_x.shape[0], B):
                xb = val_x[i:i + B]
                m = make_span_mask(xb.shape[0], L, mask_ratio, mask_span, device=dev, generator=vgen)
                tot += float(_objective_loss(net, xb, m, cfg, vgen)) * xb.shape[0]
                n += xb.shape[0]
        net.train()
        return tot / max(n, 1)

    best_state, best_val, bad = None, float("inf"), 0
    run_loss, run_n = 0.0, 0
    net.train()
    for step in range(steps):
        xb = _sample_windows(Xt, train_seg, L, B, gen)
        mask = make_span_mask(B, L, mask_ratio, mask_span, device=dev, generator=gen)
        loss = _objective_loss(net, xb, mask, cfg, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        run_loss += float(loss.detach()); run_n += 1
        history["steps_done"] = step + 1

        last = step == steps - 1
        out_of_time = time_budget_s is not None and time.perf_counter() - t_start > time_budget_s
        if (step + 1) % eval_every == 0 or last or out_of_time:
            v = validate()
            tr = run_loss / max(run_n, 1); run_loss, run_n = 0.0, 0
            history["step"].append(step + 1); history["train_loss"].append(tr)
            history["val_loss"].append(v); history["lr"].append(opt.param_groups[0]["lr"])
            if progress:
                progress(step + 1, steps, tr, v)
            if log:
                vs = "" if math.isnan(v) else f"  val={v:.4f}"
                log(f"  step {step + 1:5d}/{steps}  loss={tr:.4f}{vs}")
            if not math.isnan(v):
                if v < best_val - 1e-3 * abs(best_val if np.isfinite(best_val) else 1.0):
                    best_val, bad = v, 0
                    history["best_step"] = step + 1
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience and not last:     # out of patience before the budget
                        history["stopped_early"] = True
                        if log:
                            log(f"  early stop at step {step + 1} (best step {history['best_step']})")
                        break
            if out_of_time:
                if log:
                    log(f"  time budget {time_budget_s:.0f}s reached at step {step + 1}")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
        history["best_val"] = best_val
    else:
        history["best_step"] = history["steps_done"]
    history["seconds"] = time.perf_counter() - t_start
    if log:
        log(f"  trained {history['steps_done']} steps in {history['seconds']:.1f}s "
            f"({history['n_params'] / 1e3:.0f}k params, objective={objective})")
    return net, history


# --------------------------------------------------------------------------- #
# inference                                                                    #
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def embed_sequence(net, X, emb_dim=None, win_len=256, device="auto", batch=None,
                   overlap=0.5, normalize=False):
    """Per-frame embedding Z [T, E] of a whole recording X [C, T].

    Windows overlap by ``overlap`` and are blended with a positive taper
    (Hann for acausal models, a rising ramp for causal ones, so each frame is
    dominated by the windows where it has the most context). This removes the
    seams that non-overlapping windows leave every ``win_len`` frames.
    ``emb_dim`` is inferred from the network and kept only for compatibility.
    """
    dev = resolve_device(device)
    net.eval()
    X = np.ascontiguousarray(X, dtype=np.float32)
    C, T = X.shape
    L = int(min(win_len, T))
    pad_to = None
    if L < 32 and T < win_len:                        # very short: pad like v0.1
        pad_to, L = win_len, win_len
        Xp = np.zeros((C, win_len), np.float32); Xp[:, :T] = X; X = Xp
    Tp = X.shape[1]
    hop = max(1, int(round(L * (1.0 - overlap))))
    starts = list(range(0, Tp - L + 1, hop))
    if starts[-1] != Tp - L:
        starts.append(Tp - L)
    causal = bool(getattr(getattr(net, "cfg", None), "causal", False))
    w = (np.linspace(0.05, 1.0, L) if causal else np.hanning(L + 2)[1:-1]).astype(np.float32)
    if batch is None:                                  # channel-token models are memory hungry
        batch = max(1, min(64, 4096 // max(1, C))) if getattr(net, "channel_tokens", False) else 64
    Xt = torch.from_numpy(X).to(dev)
    acc = None
    wsum = np.zeros(Tp, np.float32)
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        xb = torch.stack([Xt[:, s:s + L] for s in ss], dim=0)
        z = net.encode(xb).float().cpu().numpy()      # [b, E, L]
        if acc is None:
            acc = np.zeros((Tp, z.shape[1]), np.float32)
        for j, s0 in enumerate(ss):
            acc[s0:s0 + L] += z[j].T * w[:, None]
            wsum[s0:s0 + L] += w
    Z = acc / wsum[:, None]
    if pad_to is not None:
        Z = Z[:T]
    if normalize:
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return Z.astype(np.float32)

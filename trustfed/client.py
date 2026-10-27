import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from .models import get_model, get_weights, set_weights


def sqrt_inv_freq_weights(y, n_classes):
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(cnt > 0, np.sqrt(cnt.sum() / (n_classes * np.maximum(cnt, 1))), 1.0)
    w = w / w[cnt > 0].mean()
    return np.clip(w, 0.1, 20.0).astype(np.float32)


def dp_epsilon(sigma, q, steps, delta=1e-5):
    if steps == 0:
        return 0.0
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)]
    return float(acc.get_epsilon(delta=delta))


def calibrate_sigma(n, cfg):
    from opacus.accountants.utils import get_noise_multiplier
    bs = min(cfg["batch_size"], n)
    q = bs / n
    steps = cfg["local_epochs"] * int(np.ceil(n / bs)) * cfg["n_rounds"]
    return float(get_noise_multiplier(target_epsilon=cfg["dp_target_eps"],
                                      target_delta=cfg.get("dp_delta", 1e-5),
                                      sample_rate=q, steps=steps, accountant="rdp",
                                      epsilon_tolerance=0.01))


def local_train(global_w, X, y, cfg, n_features, n_classes, class_w, use_dp, mu,
                dp_state=None, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = get_model(n_features, n_classes, cfg["arch"], cfg["hidden"], cfg["n_blocks"],
                      dp=use_dp).to(device)
    set_weights(model, global_w)
    gparams = [p.detach().clone() for p in model.parameters()]
    n = len(y)
    bs = min(cfg["batch_size"], n)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    cw = torch.tensor(class_w, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    t0 = time.time()
    sigma = (dp_state or {}).get("sigma", cfg["dp_sigma"])
    if use_dp:
        from opacus import PrivacyEngine
        dl = DataLoader(ds, batch_size=bs, shuffle=True)
        pe = PrivacyEngine(accountant="rdp", secure_mode=False)
        model.train()
        model, opt, dl = pe.make_private(module=model, optimizer=opt, data_loader=dl,
                                         noise_multiplier=sigma,
                                         max_grad_norm=cfg["dp_clip"],
                                         poisson_sampling=True)
    else:
        sw = np.zeros(n, np.float32)
        for c, k in zip(*np.unique(y, return_counts=True)):
            sw[y == c] = 1 / np.sqrt(k)
        g = torch.Generator().manual_seed(seed)
        dl = DataLoader(ds, batch_size=bs, sampler=WeightedRandomSampler(
            torch.from_numpy(sw), n, replacement=True, generator=g))
    model.train()
    steps, losses = 0, []
    from contextlib import nullcontext
    mbs = cfg.get("max_physical_batch", 512)
    if use_dp and mbs and mbs < bs:


        from opacus.utils.batch_memory_manager import BatchMemoryManager
        mem = lambda: BatchMemoryManager(data_loader=dl, max_physical_batch_size=mbs, optimizer=opt)
    else:
        mem = lambda: nullcontext(dl)
    for _ in range(cfg["local_epochs"]):
        with mem() as loader:
            for xb, yb in loader:
                if len(yb) == 0:
                    continue
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = (F.cross_entropy(model(xb), yb, reduction="none") * cw[yb]).mean()
                if not use_dp and mu > 0:
                    loss = loss + mu / 2 * sum(((p - g0) ** 2).sum()
                                               for p, g0 in zip(model.parameters(), gparams))
                loss.backward()
                if not use_dp:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                opt.step()
                skipped = bool(use_dp and getattr(opt, "_is_last_step_skipped", False))
                if use_dp and mu > 0 and not skipped:
                    with torch.no_grad():
                        for p, g0 in zip(model.parameters(), gparams):
                            p.add_(-(cfg["lr"] * mu) * (p - g0))
                losses.append(float(loss.item()))
                steps += 0 if skipped else 1
    info = {"n": n, "loss": float(np.mean(losses)) if losses else 0.0, "steps": steps,
            "train_sec": round(time.time() - t0, 3), "dp": bool(use_dp)}
    if use_dp:

        dp_state["steps"] = dp_state.get("steps", 0) + cfg["local_epochs"] * len(dl)
        q = float(getattr(dl, "sample_rate", bs / n))
        dp_state["q"] = q
        info["sigma"] = sigma
        info["epsilon_cumulative"] = dp_epsilon(sigma, q, dp_state["steps"],
                                                cfg.get("dp_delta", 1e-5))
        info["q"] = q
        model = model._module
    w_out = get_weights(model)


    for p_ in model.parameters():
        if hasattr(p_, "grad_sample"):
            p_.grad_sample = None
        p_.grad = None
    del model, opt, dl
    if use_dp:
        del pe
    import gc; gc.collect()
    return w_out, info

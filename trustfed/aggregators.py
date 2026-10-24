"""
aggregators.py -- every aggregation rule operates on client DELTAS (w_i - w_t)
and returns (aggregated_delta, info). info["flagged"] lists clients the rule
rejected or zero-weighted (used for Byzantine detection metrics); None means
the rule makes no client-level decision (median, trimmed mean, FedAvg).
"""
import numpy as np


def _stack(ups):
    return np.stack([u["delta"] for u in ups])


def fedavg(ups, **_):
    n = np.array([u["n"] for u in ups], float)
    return (_stack(ups) * (n / n.sum())[:, None]).sum(0), {"flagged": None}


def weighted(ups, weights, **_):
    w = np.array([weights[u["cid"]] for u in ups])
    return (_stack(ups) * (w / w.sum())[:, None]).sum(0), {}


def _krum_scores(D, f):
    n = len(D)
    sq = np.sum(D ** 2, 1)
    dist = np.maximum(sq[:, None] + sq[None, :] - 2 * D @ D.T, 0)
    m = max(1, n - f - 2)
    return np.array([np.sort(np.delete(dist[i], i))[:m].sum() for i in range(n)])


def krum(ups, f, **_):
    D = _stack(ups); sc = _krum_scores(D, f); i = int(np.argmin(sc))
    return D[i], {"flagged": [u["cid"] for j, u in enumerate(ups) if j != i]}


def multikrum(ups, f, **_):
    D = _stack(ups); sc = _krum_scores(D, f)
    keep = np.argsort(sc)[:max(1, len(ups) - f)]
    return D[keep].mean(0), {"flagged": [u["cid"] for j, u in enumerate(ups) if j not in keep]}


def median(ups, **_):
    return np.median(_stack(ups), 0), {"flagged": None}


def trimmed_mean(ups, f, **_):
    D = np.sort(_stack(ups), 0); k = min(f, (len(ups) - 1) // 2)
    return D[k:len(ups) - k].mean(0), {"flagged": None}


def flame(ups, rng, noise_lambda=0.001, **_):
    """FLAME (Nguyen et al. 2022): HDBSCAN on cosine distance, norm clipping, noise."""
    from sklearn.cluster import HDBSCAN
    D = _stack(ups); n = len(D)
    Nn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
    dist = np.clip(1 - Nn @ Nn.T, 0, 2).astype(np.float64)
    lab = HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1, metric="precomputed",
                  allow_single_cluster=True).fit_predict(dist)
    if (lab >= 0).any():
        big = np.bincount(lab[lab >= 0]).argmax(); adm = np.where(lab == big)[0]
    else:
        adm = np.arange(n)
    norms = np.linalg.norm(D, axis=1); S = np.median(norms)
    clipped = D[adm] * np.minimum(1, S / np.clip(norms[adm], 1e-12, None))[:, None]
    agg = clipped.mean(0) + rng.normal(0, noise_lambda * S, size=D.shape[1])
    return agg, {"flagged": [u["cid"] for j, u in enumerate(ups) if j not in adm]}


def fltrust(ups, server_delta, **_):
    """FLTrust (Cao et al. 2021): ReLU-cosine trust + magnitude normalisation."""
    D = _stack(ups); g = server_delta; gn = np.linalg.norm(g)
    if gn == 0:
        return D.mean(0), {"flagged": []}
    nd = np.linalg.norm(D, axis=1)
    ts = np.maximum(0, D @ g / np.clip(nd * gn, 1e-12, None))
    if ts.sum() == 0:
        return np.zeros_like(g), {"flagged": [u["cid"] for u in ups]}
    Dn = D * (gn / np.clip(nd, 1e-12, None))[:, None]
    return (Dn * ts[:, None]).sum(0) / ts.sum(), {"flagged": [u["cid"] for j, u in enumerate(ups) if ts[j] == 0]}


def foolsgold(ups, history, **_):
    """FoolsGold (Fung et al. 2020) with pardoning and logit, on cumulative history."""
    H = np.stack([history[u["cid"]] for u in ups]); n = len(H)
    if n == 1:
        return ups[0]["delta"], {"flagged": []}
    Hn = H / np.clip(np.linalg.norm(H, axis=1, keepdims=True), 1e-12, None)
    cs = Hn @ Hn.T - np.eye(n)
    mx = cs.max(1)
    for i in range(n):
        for j in range(n):
            if i != j and mx[i] < mx[j] and mx[j] > 0:
                cs[i, j] *= mx[i] / mx[j]
    wv = np.clip(1 - cs.max(1), 0, 1)
    wv = wv / wv.max() if wv.max() > 0 else np.ones(n)
    wv[wv == 1] = 0.99
    with np.errstate(divide="ignore"):
        wv = np.log(wv / (1 - wv)) + 0.5
    wv[~np.isfinite(wv) & (wv > 0)] = 1; wv[~np.isfinite(wv)] = 0
    wv = np.clip(wv, 0, 1)
    D = _stack(ups)
    agg = (D * wv[:, None]).sum(0) / wv.sum() if wv.sum() > 0 else np.zeros(D.shape[1])
    return agg, {"flagged": [u["cid"] for j, u in enumerate(ups) if wv[j] < 0.01]}


RULES = {"fedavg": fedavg, "krum": krum, "multikrum": multikrum, "median": median,
         "trimmed_mean": trimmed_mean, "flame": flame, "fltrust": fltrust,
         "foolsgold": foolsgold}

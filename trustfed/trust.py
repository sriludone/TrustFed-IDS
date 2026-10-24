"""
trust.py -- dynamic trust scoring (manuscript Methods, Eq. 2-5).

Corrections relative to the original trust_scoring.py / server.py:
  * KL signal: the original compared each client's mean prediction with a
    UNIFORM distribution, so honest clients holding mostly-benign data looked
    like label-flippers. Here the SERVER computes the divergence itself between
    the client's model and the global model on a data-free probe set (uniform
    samples in the min-max-scaled feature box). Consequences:
      - it measures behavioural drift on identical inputs, not label skew;
      - it is computed from the submitted weights, so a client cannot spoof it;
      - it is post-processing of DP-protected weights, so it costs no privacy
        (the old design released predictions on private data outside the DP
        accounting).
  * Flags are decided over the clients that PARTICIPATED this round only.
  * Safety guard does what the paper says: if >= guard_frac of participants
    fall below the threshold, that round uses equal (neutral) weights.
  * A floor of 0.01 on the aggregation weight is applied and must be stated.
Steady state (constant signals): s* = (a*cs - b*kl)/(1-g) = 0.6*cs - 0.4*kl.
"""
import numpy as np


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))


def mean_kl(p, q, eps=1e-8):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


class TrustScorer:
    def __init__(self, n_clients, alpha=0.3, beta=0.2, gamma=0.5, kappa=5.0,
                 threshold=0.20, guard_frac=0.8, floor=0.01, init=1.0,
                 use_cos=True, use_kl=True, use_ema=True, flag_rule="fixed", mad_k=1.5):
        self.s = {i: float(init) for i in range(n_clients)}
        self.a, self.b, self.g, self.k = alpha, beta, gamma, kappa
        self.thr, self.guard, self.floor = threshold, guard_frac, floor
        self.use_cos, self.use_kl, self.use_ema = use_cos, use_kl, use_ema
        self.flag_rule, self.mad_k = flag_rule, mad_k

    def step(self, deltas, global_step, kl_raw):
        """deltas: {cid: w_i - w_t}; global_step: w_t - w_{t-1}; kl_raw: {cid: mean KL}."""
        signals = {}
        for cid, d in deltas.items():
            cs = (1 + cosine(d, global_step)) / 2 if self.use_cos else 0.5
            kl = min(kl_raw.get(cid, 0.0) / self.k, 1.0) if self.use_kl else 0.0
            prev = self.s[cid] if self.use_ema else 0.0
            g = self.g if self.use_ema else 0.0
            self.s[cid] = float(np.clip(self.a * cs - self.b * kl + g * prev, 0, 1))
            signals[cid] = {"cs": round(cs, 5), "kl": round(kl, 5), "s": round(self.s[cid], 5)}
        part = list(deltas)
        if self.flag_rule == "mad":   # robust, data-driven: below median - k*MAD of this round
            v = np.array([self.s[c] for c in part]); med = np.median(v)
            mad = np.median(np.abs(v - med)) * 1.4826
            cut = med - self.mad_k * mad if mad > 0 else -1
            low = [c for c in part if self.s[c] < cut]
        else:
            low = [c for c in part if self.s[c] < self.thr]
        guard = len(low) >= self.guard * len(part)
        if guard:
            w = {c: 1.0 for c in part}
            flagged = []
        else:
            w = {c: max(self.s[c], self.floor) for c in part}
            flagged = low
        tot = sum(w.values())
        return {c: v / tot for c, v in w.items()}, flagged, guard, signals

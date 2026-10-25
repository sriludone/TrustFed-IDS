"""
agent.py -- local autonomous-response agent (Double DQN) and its evaluation.

Why not DDPG: the action set {BLOCK, QUARANTINE, ALERT, THROTTLE, PASS} is
discrete; DDPG is for continuous actions. The original agent passed the actor
output through argmax/one-hot, which has no gradient, so the actor never
learned; it was also federated (contradicting the paper) and never called.

Setting (state this in the paper):
  * Each client trains its own agent (not federated) on flows from its
    validation split, scored by the final global detector. Agents are tested
    on episodes drawn from the global TEST set (never seen by the agent).
  * Episodes are bursty: a two-state process alternates benign periods with
    bursts of a single attack class, so the "recent attack rate" feature is
    informative. Actions do not change the next flow, so the task is a
    contextual bandit; gamma defaults to 0 and can be raised (cfg).
  * Reward = paper's r_detect + r_action. The paper's r_privacy term is
    dropped: actions do not consume DP budget, so it is constant.
Ground truth "appropriate" actions: benign->PASS, low severity->ALERT/THROTTLE,
high severity->BLOCK/QUARANTINE (severity.py).
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["BLOCK", "QUARANTINE", "ALERT", "THROTTLE", "PASS"]
BLOCK, QUAR, ALERT, THROT, PASS = range(5)
APPROPRIATE = {0: {PASS}, 1: {ALERT, THROT}, 2: {BLOCK, QUAR}}


def reward(action, sev):
    attack = sev > 0
    act = action != PASS
    r = {(True, True): 10, (False, True): -15, (True, False): -5, (False, False): 1}[(attack, act)]
    if action in APPROPRIATE[sev] and sev > 0:
        r += 5
    if sev == 0 and action in (BLOCK, QUAR):
        r -= 8
    if sev == 2 and action in (ALERT, THROT):
        r -= 4
    return float(r)


def state_features(P, sev_vec, benign_idx, trust, hist_rate):
    sev_vec = np.asarray(sev_vec)
    p_att = 1 - P[:, benign_idx] if benign_idx is not None else np.ones(len(P))
    pmax = P.max(1)
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1) / np.log(P.shape[1])
    p_low = P[:, sev_vec == 1].sum(1); p_high = P[:, sev_vec == 2].sum(1)
    psev = sev_vec[P.argmax(1)]
    oh = np.eye(3)[psev]
    return np.column_stack([p_att, pmax, ent, p_low, p_high, (P * sev_vec).sum(1) / 2, oh,
                            np.full(len(P), trust), hist_rate]).astype(np.float32)


def make_episodes(y, sev_vec, n_ep, length, rng, p_start=0.05, mean_burst=40):
    """Bursty index sequences over a labelled pool."""
    sev_vec = np.asarray(sev_vec)
    ben = np.where(sev_vec[y] == 0)[0]
    att_by_c = {c: np.where(y == c)[0] for c in np.unique(y) if sev_vec[c] > 0}
    att_cls = [c for c, v in att_by_c.items() if len(v)]
    eps = []
    for _ in range(n_ep):
        seq, burst, cls = [], 0, None
        for _ in range(length):
            if burst == 0 and att_cls and rng.random() < p_start:
                cls = att_cls[rng.integers(len(att_cls))]; burst = int(rng.geometric(1 / mean_burst))
            if burst > 0 and cls is not None:
                seq.append(rng.choice(att_by_c[cls])); burst -= 1
            else:
                seq.append(rng.choice(ben) if len(ben) else rng.integers(len(y)))
        eps.append(np.array(seq))
    return eps


def _states_for_episode(P_ep, sev_vec, b, trust, window=5):
    pred_att = (np.asarray(sev_vec)[P_ep.argmax(1)] > 0).astype(float)
    hist = np.array([pred_att[max(0, i - window):i].mean() if i else 0.0 for i in range(len(P_ep))])
    return state_features(P_ep, sev_vec, b, trust, hist)


class QNet(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 5))

    def forward(self, s):
        return self.f(s)


def train_dqn(episodes_S, episodes_sev, cfg, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    d = episodes_S[0].shape[1]
    q, qt = QNet(d), QNet(d); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.get("agent_lr", 1e-3))
    gamma, tau, bs = cfg.get("agent_gamma", 0.0), cfg.get("agent_tau", 0.005), 64
    cap = cfg.get("agent_buffer", 50_000)
    B_s = np.zeros((cap, d), np.float32); B_a = np.zeros(cap, np.int64); B_r = np.zeros(cap, np.float32)
    B_s2 = np.zeros((cap, d), np.float32); B_d = np.zeros(cap, np.float32)
    ptr = full = 0
    total = sum(len(s) for s in episodes_S); step = 0; ep_rewards = []
    for S, sev in zip(episodes_S, episodes_sev):
        R = 0.0
        for t in range(len(S)):
            eps = max(0.05, 1 - step / (0.6 * total))
            if rng.random() < eps:
                a = int(rng.integers(5))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(S[t:t + 1])).argmax())
            r = reward(a, int(sev[t])); R += r
            done = float(t == len(S) - 1)
            B_s[ptr], B_a[ptr], B_r[ptr] = S[t], a, r
            B_s2[ptr] = S[min(t + 1, len(S) - 1)]; B_d[ptr] = done
            ptr = (ptr + 1) % cap; full = min(full + 1, cap); step += 1
            if full >= bs:
                i = rng.integers(0, full, bs)
                s, a_, r_, s2, d_ = map(torch.from_numpy, (B_s[i], B_a[i], B_r[i], B_s2[i], B_d[i]))
                with torch.no_grad():
                    a2 = q(s2).argmax(1, keepdim=True)
                    tgt = r_ + gamma * (1 - d_) * qt(s2).gather(1, a2).squeeze(1)
                loss = F.smooth_l1_loss(q(s).gather(1, a_[:, None]).squeeze(1), tgt)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10); opt.step()
                with torch.no_grad():
                    for p, pt in zip(q.parameters(), qt.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)
        ep_rewards.append(R / len(S))
    return q, ep_rewards


def policy_actions(name, S, P, sev_vec, b, q=None, true_sev=None, rng=None):
    if name == "dqn":
        with torch.no_grad():
            return q(torch.from_numpy(S)).argmax(1).numpy()
    if name == "threshold_rule":                      # the paper's ablation A1 rule
        p_att = 1 - P[:, b]
        return np.where(p_att > 0.8, BLOCK, PASS)
    if name == "severity_rule":                       # predicted class -> canonical action
        ps = np.asarray(sev_vec)[P.argmax(1)]
        return np.select([ps == 2, ps == 1], [BLOCK, ALERT], PASS)
    if name == "bayes_rule":                          # expected-reward (cost-sensitive) decision rule
        sv = np.asarray(sev_vec)
        ps = np.stack([P[:, sv == v].sum(1) for v in (0, 1, 2)], 1)          # P(severity | x)
        R = np.array([[reward(a, v) for v in (0, 1, 2)] for a in range(5)])  # reward table (5 x 3)
        return (ps @ R.T).argmax(1)
    if name == "oracle":                              # true label -> canonical action (upper bound)
        return np.select([true_sev == 2, true_sev == 1], [BLOCK, ALERT], PASS)
    return rng.integers(0, 5, len(S))                 # random


def action_metrics(acts, sev):
    acts, sev = np.asarray(acts), np.asarray(sev)
    ben, low, high = sev == 0, sev == 1, sev == 2
    ok = np.array([a in APPROPRIATE[s] for a, s in zip(acts, sev)])
    disrupt = np.isin(acts, [BLOCK, QUAR])
    return {"action_accuracy": float(ok.mean()),
            "benign_disruption_rate": float(disrupt[ben].mean()) if ben.any() else None,
            "benign_any_intervention": float((acts[ben] != PASS).mean()) if ben.any() else None,
            "high_containment_rate": float(disrupt[high].mean()) if high.any() else None,
            "low_appropriate_rate": float(np.isin(acts[low], [ALERT, THROT]).mean()) if low.any() else None,
            "missed_attack_rate": float((acts[~ben] == PASS).mean()) if (~ben).any() else None,
            "mean_reward": float(np.mean([reward(int(a), int(s)) for a, s in zip(acts, sev)])),
            "action_counts": np.bincount(acts, minlength=5).tolist()}


def run_agents(fed, best_w, cfg, seed, trust_scores=None, verbose=True):
    """Returns the agent evaluation as a dict (stored inside the run's JSON)."""
    from .models import get_model, set_weights, predict_proba
    nf, nc, b, sevv = fed["n_features"], fed["n_classes"], fed["benign_idx"], np.asarray(fed["severity"])
    det = get_model(nf, nc, cfg["arch"], cfg["hidden"], cfg["n_blocks"]); set_weights(det, best_w)
    rng = np.random.default_rng(seed + 99)
    L, n_tr, n_te = cfg.get("agent_ep_len", 500), cfg.get("agent_train_episodes", 200), cfg.get("agent_test_episodes", 20)
    te_eps = make_episodes(fed["y_test"], sevv, n_te, L, np.random.default_rng(seed + 5))
    P_test = predict_proba(det, fed["X_test"])
    per_client, curves = [], {}
    for c in fed["clients"]:
        Xa = c["X_val"] if len(c["y_val"]) >= 200 else np.concatenate([c["X_val"], c["X_train"]])
        ya = c["y_val"] if len(c["y_val"]) >= 200 else np.concatenate([c["y_val"], c["y_train"]])
        if len(ya) < 50 or (sevv[ya] > 0).sum() == 0 or b is None:
            continue
        trust = float(trust_scores.get(str(c["id"]), 1.0)) if trust_scores else 1.0
        Pa = predict_proba(det, Xa)
        tr_eps = make_episodes(ya, sevv, n_tr, L, rng)
        S_tr = [_states_for_episode(Pa[e], sevv, b, trust) for e in tr_eps]
        q, curve = train_dqn(S_tr, [sevv[ya[e]] for e in tr_eps], cfg, seed + c["id"])
        curves[str(c["id"])] = curve
        res = {"client": c["id"], "trust": trust}
        for pol in ("dqn", "bayes_rule", "threshold_rule", "severity_rule", "random", "oracle"):
            A_, Sv = [], []
            for e in te_eps:
                S = _states_for_episode(P_test[e], sevv, b, trust)
                A_.append(policy_actions(pol, S, P_test[e], sevv, b, q, sevv[fed["y_test"][e]],
                                         np.random.default_rng(seed)))
                Sv.append(sevv[fed["y_test"][e]])
            res[pol] = action_metrics(np.concatenate(A_), np.concatenate(Sv))
        x1 = torch.from_numpy(fed["X_test"][:1]); s1 = torch.from_numpy(S[:1])
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(200):
                q(s1)
            res["agent_ms"] = (time.perf_counter() - t0) / 200 * 1000
            t0 = time.perf_counter()
            for _ in range(200):
                det(x1); q(s1)
            res["end_to_end_ms"] = (time.perf_counter() - t0) / 200 * 1000
        per_client.append(res)
        if verbose:
            print(f"  [agent c{c['id']}] dqn acc={res['dqn']['action_accuracy']:.3f} "
                  f"sev_rule={res['severity_rule']['action_accuracy']:.3f} "
                  f"thr={res['threshold_rule']['action_accuracy']:.3f}", flush=True)
    keys = ["action_accuracy", "benign_disruption_rate", "benign_any_intervention", "high_containment_rate",
            "low_appropriate_rate", "missed_attack_rate", "mean_reward"]
    agg = {pol: {k: [float(np.mean([r[pol][k] for r in per_client if r[pol][k] is not None])),
                     float(np.std([r[pol][k] for r in per_client if r[pol][k] is not None]))]
                 for k in keys} for pol in ("dqn", "bayes_rule", "threshold_rule", "severity_rule", "random", "oracle")}
    out = {"dataset": fed["dataset"], "seed": seed, "n_agents": len(per_client),
           "summary_mean_sd": agg, "per_client": per_client,
           "agent_ms_mean": float(np.mean([r["agent_ms"] for r in per_client])),
           "end_to_end_ms_mean": float(np.mean([r["end_to_end_ms"] for r in per_client])),
           "severity_map": dict(zip(fed["class_names"], fed["severity"])),
           "settings": {"gamma": cfg.get("agent_gamma", 0.0), "episode_len": L,
                        "train_episodes": n_tr, "test_episodes": n_te}}
    # reward curves downsampled to <= 50 points per client agent
    out["reward_curves"] = {c: [round(float(x), 3) for x in v[:: max(1, len(v) // 50)]]
                            for c, v in curves.items()}
    return out

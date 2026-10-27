"""
runner.py -- one federated training loop for EVERY FL method.

Identical across methods: data split, client partition, per-round client
selection, local training procedure, evaluation. Only the aggregation rule,
DP on/off and the FedProx mu differ (see PRESETS).

Model selection: federated validation (global model evaluated on the union of
client validation splits). Early stopping and the reported round use VALIDATION
balanced accuracy; test metrics are reported at that round. The original code
selected the best round on the TEST set.
"""
import os, json, time, copy
import numpy as np
import torch

from .models import get_model, get_weights, set_weights, count_parameters, predict_proba
from .client import local_train, sqrt_inv_freq_weights, dp_epsilon, calibrate_sigma
from .metrics import evaluate
from .trust import TrustScorer, mean_kl
from .aggregators import RULES, weighted
from . import attacks as A
from .audit import AuditLedger, verify_chain, tamper_suite
from .explain import shap_for_predicted

PRESETS = {
    # proposed method
    "trustfed":          dict(aggregator="trust",  use_dp=True,  mu=0.01),
    # ablations (detection layer only)
    "abl_no_trust":      dict(aggregator="fedavg", use_dp=True,  mu=0.01),
    "abl_uniform":       dict(aggregator="uniform", use_dp=True, mu=0.01),   # equal weights, no trust
    "abl_no_prox":       dict(aggregator="trust",  use_dp=True,  mu=0.0),
    "abl_no_dp":         dict(aggregator="trust",  use_dp=False, mu=0.01),
    "abl_no_kl":         dict(aggregator="trust",  use_dp=True,  mu=0.01, trust_use_kl=False),
    "abl_no_cos":        dict(aggregator="trust",  use_dp=True,  mu=0.01, trust_use_cos=False),
    "abl_no_ema":        dict(aggregator="trust",  use_dp=True,  mu=0.01, trust_use_ema=False),
    # baselines
    "fedavg":            dict(aggregator="fedavg", use_dp=False, mu=0.0),
    "fedprox":           dict(aggregator="fedavg", use_dp=False, mu=0.01),
    "dp_fedavg":         dict(aggregator="fedavg", use_dp=True,  mu=0.0),
    "fltrust":           dict(aggregator="fltrust", use_dp=False, mu=0.0),
    "foolsgold":         dict(aggregator="foolsgold", use_dp=False, mu=0.0),
    "krum":              dict(aggregator="krum",   use_dp=False, mu=0.0),
    "multikrum":         dict(aggregator="multikrum", use_dp=False, mu=0.0),
    "median":            dict(aggregator="median", use_dp=False, mu=0.0),
    "trimmed_mean":      dict(aggregator="trimmed_mean", use_dp=False, mu=0.0),
    "flame":             dict(aggregator="flame",  use_dp=False, mu=0.0),
}


def _probe(n_features, n=512, seed=12345):
    return np.random.default_rng(seed).uniform(0, 1, (n, n_features)).astype(np.float32)


def run_fl(fed, cfg, method, seed, attack="none", byz_frac=0.0, flip_rate=1.0,
           attack_start=1, attack_scale=5.0, force_dp=None, audit=False, audit_file=None, verbose=True):
    """Returns (summary, best_weights, compact_round_log). Writes nothing except, when
    audit_file is given, ONE compressed file with the run's SHAP vectors."""
    t_run = time.time()
    p = dict(PRESETS[method])
    use_dp = p["use_dp"] if force_dp is None else bool(force_dp)
    mu = p["mu"]
    dev = cfg.get("device", "cpu")
    torch.manual_seed(seed)
    nf, nc, N = fed["n_features"], fed["n_classes"], len(fed["clients"])
    bidx = fed["benign_idx"]
    clients = fed["clients"]
    m = max(1, int(round(cfg["participation"] * N)))
    n_mal = int(round(byz_frac * N)) if attack != "none" else 0
    mal = set(range(n_mal))
    f_est = max(1, int(round(byz_frac * m))) if n_mal else max(1, int(0.2 * m))

    # training-pool class histogram (aggregate counts; never the test set)
    ytr_all = np.concatenate([c["y_train"] for c in clients])
    class_w = sqrt_inv_freq_weights(ytr_all, nc)

    # poisoned labels prepared once per malicious client (local RNG)
    y_poison = {}
    if A.needs_flipped_data(attack):
        for c in mal:
            y_poison[c] = A.label_flip(clients[c]["y_train"], bidx, flip_rate,
                                       np.random.default_rng(seed * 100 + c))

    model = get_model(nf, nc, cfg["arch"], cfg["hidden"], cfg["n_blocks"], dp=False)
    w = get_weights(model); w_prev = w.copy()
    scorer = TrustScorer(N, cfg["trust_alpha"], cfg["trust_beta"], cfg["trust_gamma"],
                         cfg["trust_kappa"], cfg["trust_threshold"],
                         use_cos=p.get("trust_use_cos", True), use_kl=p.get("trust_use_kl", True),
                         use_ema=p.get("trust_use_ema", True),
                         flag_rule=cfg.get("trust_flag_rule", "fixed"), mad_k=cfg.get("trust_mad_k", 1.5))
    probe = _probe(nf)
    history = {i: np.zeros_like(w) for i in range(N)}
    dp_state = {i: {} for i in range(N)}
    if use_dp and cfg.get("dp_target_eps"):
        for c in clients:
            if len(c["y_train"]):
                dp_state[c["id"]]["sigma"] = calibrate_sigma(len(c["y_train"]), cfg)
        if verbose:
            s = [v["sigma"] for v in dp_state.values() if "sigma" in v]
            print(f"  DP target eps={cfg['dp_target_eps']}: per-client sigma "
                  f"min={min(s):.2f} median={np.median(s):.2f} max={max(s):.2f}")
    Xv = np.concatenate([c["X_val"] for c in clients]); yv = np.concatenate([c["y_val"] for c in clients])
    ledger = AuditLedger() if audit else None
    agg_rng = np.random.default_rng(seed + 7)
    logs, best, best_w, patience = [], -1.0, w.copy(), 0
    eval_model = get_model(nf, nc, cfg["arch"], cfg["hidden"], cfg["n_blocks"], dp=False)

    for rnd in range(1, cfg["n_rounds"] + 1):
        t0 = time.time()
        sel = sorted(np.random.default_rng(seed * 1000 + rnd).choice(N, m, replace=False).tolist())
        ups, infos = [], {}
        for cid in sel:
            c = clients[cid]
            if len(c["y_train"]) == 0:
                continue
            active = cid in mal and A.is_active(attack, rnd, attack_start)
            y_use = y_poison[cid] if (active and cid in y_poison) else c["y_train"]
            w_i, info = local_train(w, c["X_train"], y_use, cfg, nf, nc, class_w, use_dp, mu,
                                    dp_state[cid], seed=seed * 10000 + rnd * 100 + cid, device=dev)
            d = (w_i - w).astype(np.float32)
            if active and attack in ("gaussian", "sign_flip", "model_replacement"):
                d = A.poison_update(d, attack, np.random.default_rng(seed + rnd * 31 + cid),
                                    scale=attack_scale, n_participants=m)
            ups.append({"cid": cid, "delta": d, "n": info["n"]})
            infos[cid] = info
            for k in ("epsilon_cumulative",):
                if k in info:
                    infos[cid][k] = round(info[k], 4)
            if audit:
                set_weights(eval_model, w + d)
                k = min(32, len(c["X_val"]) or len(c["X_train"]))
                Xr = (c["X_val"] if len(c["X_val"]) else c["X_train"])[:k]
                Xb = c["X_train"][:32]
                sv, st, sec = shap_for_predicted(eval_model, Xr, Xb, cfg.get("shap_method", "expected_gradients"),
                                                 seed=seed + rnd)
                head, hsec = ledger.append(cid, rnd, sv, st, use_dp)
                infos[cid].update({"shap_status": st, "shap_sec": round(sec, 3),
                                   "hash_sec": round(hsec, 5)})
        if not ups:
            continue

        # ---------------- aggregation
        info_agg, weights, signals, guard = {}, None, None, False
        rule = p["aggregator"]
        if rule == "trust":
            set_weights(eval_model, w); Pg = predict_proba(eval_model, probe)
            kl = {}
            for u in ups:
                set_weights(eval_model, w + u["delta"])
                kl[u["cid"]] = mean_kl(predict_proba(eval_model, probe), Pg)
            weights, flagged, guard, signals = scorer.step({u["cid"]: u["delta"] for u in ups},
                                                           (w - w_prev), kl)
            agg, _ = weighted(ups, weights)
            info_agg["flagged"] = flagged
        elif rule == "fltrust":
            w_s, _ = local_train(w, fed["X_root"], fed["y_root"], cfg, nf, nc, class_w, False, 0.0,
                                 seed=seed + rnd, device=dev)
            agg, info_agg = RULES["fltrust"](ups, server_delta=(w_s - w))
        elif rule == "foolsgold":
            for u in ups:
                history[u["cid"]] += u["delta"]
            agg, info_agg = RULES["foolsgold"](ups, history=history)
        elif rule == "flame":
            agg, info_agg = RULES["flame"](ups, rng=agg_rng)
        else:
            agg, info_agg = RULES[rule](ups, f=f_est)
        w_prev, w = w, (w + agg).astype(np.float32)

        # ---------------- evaluation (selection on validation only)
        set_weights(eval_model, w)
        mv, _ = evaluate(predict_proba(eval_model, Xv), yv, nc, bidx)
        mt, _ = evaluate(predict_proba(eval_model, fed["X_test"]), fed["y_test"], nc, bidx)
        flagged = info_agg.get("flagged")
        part_mal = [c for c in sel if c in mal]
        byz = {}
        if n_mal and flagged is not None:
            fl = set(flagged)
            tp = len(fl & set(part_mal)); fp = len(fl - mal)
            honest = [c for c in sel if c not in mal]
            byz = {"byz_tp": tp, "byz_fp": fp, "byz_detect_rate": tp / len(part_mal) if part_mal else None,
                   "honest_false_flag_rate": fp / len(honest) if honest else None}
        r4 = lambda x: None if x is None else round(float(x), 4)
        eps_now = [v["epsilon_cumulative"] for v in infos.values() if "epsilon_cumulative" in v]
        log = {"round": rnd, "n_sel": len(sel),
               "val_balacc": r4(mv["balanced_accuracy"]), "val_f1": r4(mv["f1_macro"]),
               "test_balacc": r4(mt["balanced_accuracy"]), "test_f1": r4(mt["f1_macro"]),
               "test_acc": r4(mt["accuracy"]), "test_benign_far": r4(mt.get("benign_far")),
               "flagged": flagged, "guard": guard, "round_sec": round(time.time() - t0, 2),
               "eps_max": r4(max(eps_now)) if eps_now else None,
               "shap_sec": r4(np.mean([v["shap_sec"] for v in infos.values() if "shap_sec" in v])) if audit else None,
               **{k: r4(v) if isinstance(v, float) else v for k, v in byz.items()}}
        if signals is not None:
            log["trust"] = {str(k): [v["cs"], v["kl"], v["s"]] for k, v in signals.items()}
        logs.append(log)
        if verbose:
            print(f"  [{method}|{fed['dataset']}|s{seed}] r{rnd:3d} valBA={mv['balanced_accuracy']:.4f} "
                  f"testBA={mt['balanced_accuracy']:.4f} F1={mt['f1_macro']:.4f} "
                  f"flag={flagged if flagged is not None else '-'} {log['round_sec']}s", flush=True)
        if mv["balanced_accuracy"] > best:
            best, best_w, best_round, patience = mv["balanced_accuracy"], w.copy(), rnd, 0
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                break

    # ---------------- final report at the best VALIDATION round
    set_weights(eval_model, best_w)
    final, cm = evaluate(predict_proba(eval_model, fed["X_test"]), fed["y_test"], nc, bidx, with_report=True)
    eps = {str(k): round(dp_epsilon(v.get("sigma", cfg["dp_sigma"]), v["q"], v["steps"],
                                    cfg.get("dp_delta", 1e-5)), 4)
           for k, v in dp_state.items() if v.get("steps")}
    sig = {str(k): round(v["sigma"], 3) for k, v in dp_state.items() if "sigma" in v}
    sizes = {str(c["id"]): int(len(c["y_train"])) for c in clients}
    byz_rounds = [l for l in logs if l.get("byz_detect_rate") is not None]
    summary = {
        "dataset": fed["dataset"], "method": method, "seed": seed, "attack": attack,
        "byz_frac": byz_frac, "attack_start": attack_start, "use_dp": use_dp, "mu": mu,
        "alpha": fed["alpha"], "best_val_round": best_round, "rounds_run": len(logs),
        "best_val_balanced_accuracy": best, "test": final,
        "epsilon_per_client": eps, "sigma_per_client": sig, "dp_target_eps": cfg.get("dp_target_eps"),
        "client_train_sizes": sizes,
        "epsilon_max": max(eps.values()) if eps else None,
        "epsilon_median": float(np.median(list(eps.values()))) if eps else None,
        "byz_detect_rate_mean": float(np.mean([l["byz_detect_rate"] for l in byz_rounds])) if byz_rounds else None,
        "honest_false_flag_rate_mean": float(np.mean([l["honest_false_flag_rate"] for l in byz_rounds
                                                      if l["honest_false_flag_rate"] is not None])) if byz_rounds else None,
        "n_params": count_parameters(eval_model), "class_names": fed["class_names"],
        "dropped_classes": fed.get("dropped_classes", {}),
        "train_test_shared_feature_vectors": fed.get("train_test_shared_feature_vectors"),
        "test_class_counts": fed.get("test_class_counts"), "client_train_sizes_list": fed.get("client_sizes"),
        "severity": fed["severity"], "runtime_sec": round(time.time() - t_run, 1),
        "mean_round_sec": float(np.mean([l["round_sec"] for l in logs])),
        "config": {k: v for k, v in cfg.items() if not isinstance(v, (list, dict))},
    }
    if audit:
        summary["audit_verify"] = {str(c): verify_chain(ledger.entries[c], ledger.vectors, ledger.anchors[c])
                                   for c in ledger.entries}
        c0 = next(iter(ledger.entries))
        summary["audit_tamper_suite"] = tamper_suite(ledger.entries[c0], ledger.vectors, ledger.anchors[c0])
        summary["audit_chain"] = ledger.to_json()
        if audit_file:
            ledger.save_vectors(audit_file)
            summary["audit_vectors_file"] = os.path.basename(audit_file)
    return summary, best_w, logs

"""
run_experiments.py -- every experiment in the paper, resumable.

  python run_experiments.py --group main      [--datasets ...] [--seeds 42 123 256 512 1024]
  python run_experiments.py --group baselines
  python run_experiments.py --group byzantine [--datasets CICIDS2017]
  python run_experiments.py --group ablation
  python run_experiments.py --group alpha
  python run_experiments.py --group privacy   # epsilon in {1, 3, 8, inf}
  python run_experiments.py --group all --dry-run      # list runs, no training
  python run_experiments.py --group byzantine --shard 0/3   # split across 3 machines

Each run writes ONE file: results/<group>/<run name>.json (main runs also write one
<run name>.audit.npz). A run whose .json exists is skipped, so the campaign can be
stopped and restarted at any time. To merge machines, copy all files into the same
results/<group>/ folders. Then run
  python make_tables.py
"""
import os, sys, json, argparse, itertools, time
import yaml

from trustfed.data import prepare_federation
from trustfed.runner import run_fl
from trustfed.centralised import run_centralised
from trustfed.agent import run_agents
from trustfed.io import run_name, run_path, write_json

SEEDS5 = [42, 123, 256, 512, 1024]
SEEDS3 = [42, 123, 256]
# Exactly the experiments reported in the Scientific Reports manuscript
BASELINES = ["centralised", "fedavg", "fedprox", "dp_fedavg", "fltrust", "flame"]
ROBUST = ["trustfed", "dp_fedavg", "multikrum", "fltrust", "flame"]          # poisoning comparison
ABLATIONS = ["trustfed", "abl_uniform", "abl_no_trust", "abl_no_kl", "abl_no_cos", "abl_no_ema"]


def plan(group, cfg, datasets, seeds, byz_datasets):
    R = []  # (group, dataset, method, attack, frac, start, alpha, seed, extras)
    if group in ("main", "all"):
        for ds, s in itertools.product(datasets, seeds or SEEDS5):
            R.append(("main", ds, "trustfed", "none", 0.0, 1, cfg["alpha"], s, {"audit": True, "agent": True}))
    if group in ("baselines", "all"):
        for ds, m, s in itertools.product(datasets, BASELINES, seeds or SEEDS5):
            R.append(("baselines", ds, m, "none", 0.0, 1, cfg["alpha"], s, {}))
    if group in ("privacy", "all"):
        for ds, e, s in itertools.product(datasets, [1.0, 3.0, 8.0], seeds or SEEDS3):
            R.append(("privacy", ds, "trustfed", "none", 0.0, 1, cfg["alpha"], s, {"eps": e}))
        for ds, s in itertools.product(datasets, seeds or SEEDS3):
            R.append(("privacy", ds, "abl_no_dp", "none", 0.0, 1, cfg["alpha"], s, {"eps": "inf"}))
    if group in ("byzantine", "all"):      # 30 % label-flipping clients, all rules under DP
        for ds, m, s in itertools.product(byz_datasets, ROBUST, seeds or SEEDS3):
            R.append(("byzantine", ds, m, "label_flip", 0.3, 1, cfg["alpha"], s, {"force_dp": True}))
    if group in ("ablation", "all"):
        for ds, m, s in itertools.product(datasets, ABLATIONS, seeds or SEEDS3):
            R.append(("ablation", ds, m, "none", 0.0, 1, cfg["alpha"], s, {}))
            R.append(("ablation", ds, m, "label_flip", 0.3, 1, cfg["alpha"], s, {}))
    if group == "randomsplit":       # reviewer request: same pipeline under an ordinary random split
        for ds, m, s in itertools.product(datasets, ["trustfed", "dp_fedavg", "centralised"], seeds or SEEDS3):
            R.append(("randomsplit", ds, m, "none", 0.0, 1, cfg["alpha"], s, {"split": "random"}))
    return R


def out_path(root, r):
    g, ds, m, atk, fr, st, a, s, ex = r
    return run_path(root, g, run_name(ds, m, atk, fr, st, a, s, ex.get("eps")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--byz-datasets", nargs="*", default=["Edge-IIoTset"])
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--results", default="results")
    ap.add_argument("--device")
    ap.add_argument("--quick", action="store_true", help="5 rounds, 1 local epoch (pipeline check only)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--processed-dir")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.device:
        cfg["device"] = a.device
    if a.quick:
        cfg.update(n_rounds=5, local_epochs=1, agent_train_episodes=10, agent_test_episodes=3,
                   centralised_epochs=2)
    datasets = a.datasets or cfg["datasets"]
    runs = plan(a.group, cfg, datasets, a.seeds, a.byz_datasets)
    if a.methods:
        runs = [r for r in runs if r[2] in a.methods]
    k, n = map(int, a.shard.split("/"))
    runs = [r for i, r in enumerate(runs) if i % n == k]
    todo = [r for r in runs if not os.path.exists(out_path(a.results, r))]
    print(f"{len(runs)} runs in plan, {len(todo)} still to do")
    if a.dry_run:
        from collections import Counter
        for key, c in sorted(Counter((r[0], r[1], r[2]) for r in todo).items()):
            print(f"  {key}: {c}")
        return
    cache = {}
    for i, r in enumerate(todo, 1):
        g, ds, m, atk, fr, st, alpha, s, ex = r
        path = out_path(a.results, r)
        print(f"\n=== [{i}/{len(todo)}] {g} {ds} {m} attack={atk} frac={fr} start={st} alpha={alpha} seed={s}")
        key = (ds, alpha, s, ex.get("split", "group"))
        if key not in cache:
            cache.clear()
            cache[key] = prepare_federation(ds, cfg["n_clients"], alpha, s,
                                            root_per_class=cfg["root_per_class"],
                                            processed_dir=a.processed_dir,
                                            scaler=cfg.get("scaler", "quantile"),
                                            min_client_size=cfg.get("min_client_size", 0),
                                            min_test_per_class=cfg.get("min_test_per_class", 30),
                                            min_unique_patterns=cfg.get("min_unique_patterns", 10),
                                            split=ex.get("split", "group"))
        fed = cache[key]
        if m in ("centralised", "lstm_ids"):
            summ, _, logs = run_centralised(fed, cfg, s, arch="residual_mlp" if m == "centralised" else "bilstm")
            write_json(path, {"group": g, **summ, "rounds": logs})
            continue
        rcfg = dict(cfg)
        if "eps" in ex and ex["eps"] != "inf":
            rcfg["dp_target_eps"] = float(ex["eps"])
        audit_file = path[:-5] + ".audit.npz" if ex.get("audit") else None
        summ, w, logs = run_fl(fed, rcfg, m, s, attack=atk, byz_frac=fr, attack_start=st,
                               force_dp=ex.get("force_dp"), audit=ex.get("audit", False),
                               audit_file=audit_file)
        record = {"group": g, "eps_setting": ex.get("eps"), **summ, "rounds": logs}
        if ex.get("agent"):
            trust = {}
            for l in logs:
                for c, v in (l.get("trust") or {}).items():
                    trust[c] = v[2]
            record["agent"] = run_agents(fed, w, rcfg, s, trust_scores=trust)
        write_json(path, record)      # written last and atomically: a crash leaves no partial result


if __name__ == "__main__":
    main()

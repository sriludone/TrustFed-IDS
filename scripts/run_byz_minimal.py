"""Minimal Byzantine experiment: 30% label-flipping clients on Edge-IIoTset, all five
aggregation rules under DP, three seeds (15 runs). Results go to results/byzantine/ with
the same file names as the full byzantine group, so finished runs are reused if the full
group is ever run later.

  python scripts/run_byz_minimal.py --device cuda --shard 0/4      # one process per GPU
  python scripts/run_byz_minimal.py --dry-run
"""
import os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import yaml
import run_experiments as R
from trustfed.data import prepare_federation
from trustfed.runner import run_fl
from trustfed.io import write_json

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda"); ap.add_argument("--results", default="results")
ap.add_argument("--shard", default="0/1"); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--attacks", nargs="*", default=["label_flip"],
                help="any of: label_flip sign_flip model_replacement gaussian (all at 30%%)")
a = ap.parse_args()
cfg = yaml.safe_load(open("config.yaml")); cfg["device"] = a.device
RULES = ["trustfed", "dp_fedavg", "multikrum", "fltrust", "flame"]
runs = [r for r in R.plan("byzantine", cfg, cfg["datasets"], None, ["Edge-IIoTset"])
        if r[2] in RULES and r[3] in a.attacks and abs(r[4] - 0.3) < 1e-9 and r[5] == 1]
k, n = map(int, a.shard.split("/")); runs = [r for i, r in enumerate(runs) if i % n == k]
todo = [r for r in runs if not os.path.exists(R.out_path(a.results, r))]
print(f"{len(runs)} runs in this shard, {len(todo)} still to do")
if a.dry_run:
    for r in todo: print("  ", r[2], "seed", r[7])
    sys.exit()
cache = {}
for i, r in enumerate(todo, 1):
    g, ds, m, atk, fr, st, alpha, s, ex = r
    print(f"\n=== [{i}/{len(todo)}] {m} {ds} {atk} 30% seed={s}", flush=True)
    if (ds, s) not in cache:
        cache.clear()
        cache[(ds, s)] = prepare_federation(ds, cfg["n_clients"], alpha, s, root_per_class=cfg["root_per_class"],
                                            scaler=cfg.get("scaler", "quantile"),
                                            min_client_size=cfg.get("min_client_size", 0),
                                            min_test_per_class=cfg.get("min_test_per_class", 30),
                                            min_unique_patterns=cfg.get("min_unique_patterns", 10))
    summ, _, logs = run_fl(cache[(ds, s)], cfg, m, s, attack=atk, byz_frac=fr, attack_start=st, force_dp=True)
    write_json(R.out_path(a.results, r), {"group": g, "eps_setting": None, **summ, "rounds": logs})

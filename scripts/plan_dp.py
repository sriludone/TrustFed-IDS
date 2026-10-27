"""Predict per-client epsilon BEFORE training (no model is trained).
   python scripts/plan_dp.py --dataset CICIDS2017 --sigma 2.77 --epochs 5 --batch 512
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml
from trustfed.data import prepare_federation
from trustfed.client import dp_epsilon, calibrate_sigma

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--config", default="config.yaml")
ap.add_argument("--sigma", type=float, nargs="+"); ap.add_argument("--epochs", type=int, nargs="+")
ap.add_argument("--batch", type=int, nargs="+"); ap.add_argument("--rounds", type=int, default=None)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
fed = prepare_federation(a.dataset, cfg["n_clients"], cfg["alpha"], a.seed, root_per_class=cfg["root_per_class"],
                         min_client_size=cfg.get("min_client_size", 0),
                         min_unique_patterns=cfg.get("min_unique_patterns", 10))
R = a.rounds or cfg["n_rounds"]; part = cfg["participation"]
sizes = np.array([len(c["y_train"]) for c in fed["clients"]])
print(f"{a.dataset}: client train sizes min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}")
print(f"assuming each client participates in {part:.0%} of {R} rounds (upper bound: no early stop)")
print(f"{'sigma':>6} {'epochs':>6} {'batch':>6} | eps(min) eps(median) eps(max)")
for s in a.sigma or [cfg["dp_sigma"]]:
    for e in a.epochs or [cfg["local_epochs"]]:
        for b in a.batch or [cfg["batch_size"]]:
            eps = [dp_epsilon(s, min(b / n, 1), int(np.ceil(n / b)) * e * int(round(part * R)), cfg["dp_delta"])
                   for n in sizes if n > 0]
            print(f"{s:6.2f} {e:6d} {b:6d} | {min(eps):8.2f} {np.median(eps):10.2f} {max(eps):8.2f}")

if cfg.get("dp_target_eps"):
    print(f"\nPer-client sigma needed for worst-case eps={cfg['dp_target_eps']} "
          f"(epochs={cfg['local_epochs']}, batch={cfg['batch_size']}, rounds={cfg['n_rounds']}):")
    s = sorted((calibrate_sigma(int(n), cfg), int(n)) for n in sizes if n > 0)
    print(f"  smallest client n={min(sizes)}: sigma={max(x[0] for x in s):.2f}")
    print(f"  median sigma={np.median([x[0] for x in s]):.2f}, largest-client sigma={min(x[0] for x in s):.2f}")
    print("  Rule of thumb: sigma above ~3 for most clients will cost noticeable macro-F1;")
    print("  then try fewer local epochs, a larger batch, or fewer rounds.")

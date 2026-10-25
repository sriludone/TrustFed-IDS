"""Quantifies train/test leakage WITHOUT training (laptop, minutes): for each dataset and
seed, the share of test rows that have an identical feature vector in the training data
under an ordinary random split versus the group-aware split used in the paper.
   python scripts/leak_stats.py --datasets Edge-IIoTset CICIDS2017 --seeds 42 123 256"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from trustfed.data import prepare_federation
ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=["Edge-IIoTset", "CICIDS2017"])
ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 256])
a = ap.parse_args(); cfg = yaml.safe_load(open("config.yaml")); out = []
for ds in a.datasets:
    for s in a.seeds:
        for sp in ("random", "group"):
            f = prepare_federation(ds, cfg["n_clients"], cfg["alpha"], s, root_per_class=cfg["root_per_class"],
                                   min_client_size=cfg.get("min_client_size", 0), split=sp,
                                   min_unique_patterns=cfg.get("min_unique_patterns", 10))
            n = sum(f["test_class_counts"]); tw = f["test_rows_with_train_twin"]
            out.append({"dataset": ds, "seed": s, "split": sp, "test_rows": n, "with_train_twin": tw,
                        "pct": round(100 * tw / n, 2), "client_sizes": f["client_sizes"]})
            print(f"{ds:13s} seed={s:<5d} {sp:6s}: {tw:>8,}/{n:,} test rows ({100*tw/n:5.1f}%) have an identical training row")
json.dump(out, open("leak_stats.json", "w"), indent=1)
print("saved leak_stats.json")

import os, json, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=["Edge-IIoTset", "CICIDS2017"])
ap.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 256])
ap.add_argument("--processed-dir", default=os.path.join("data", "processed"))
ap.add_argument("--test-size", type=float, default=0.15)
ap.add_argument("--min-test-per-class", type=int, default=30)
ap.add_argument("--min-unique-patterns", type=int, default=10)
a = ap.parse_args()


def load(ds):
    df = pd.read_csv(os.path.join(a.processed_dir, f"{ds}_processed.csv"), low_memory=False)
    feat = [c for c in df.columns if c != "label"]
    key = pd.util.hash_pandas_object(df[feat], index=False).values
    lab = df["label"].astype(str).values

    n_pat = pd.Series(key).groupby(lab).nunique()
    keep_cls = set(n_pat[n_pat >= a.min_unique_patterns].index)
    m = np.isin(lab, list(keep_cls))
    return key[m], lab[m], sorted(set(n_pat.index) - keep_cls)


def split(key, lab, seed, mode):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(lab), bool)
    for c in np.unique(lab):
        rows = np.where(lab == c)[0]
        if len(rows) < 2:
            continue
        target = max(a.test_size * len(rows), min(a.min_test_per_class, len(rows) // 3))
        if mode == "random":
            n = int(round(target)); n = max(1, min(n, len(rows) - 1))
            test[rng.permutation(rows)[:n]] = True
            continue
        g, inv, sizes = np.unique(key[rows], return_inverse=True, return_counts=True)
        if len(g) < 2:
            continue
        limit = max(1.5 * target, target + a.min_test_per_class)
        taken, chosen = 0, []
        for gi in rng.permutation(len(g)):
            if taken >= target or len(chosen) >= len(g) - 1:
                break
            if taken + sizes[gi] > limit:
                continue
            chosen.append(gi); taken += sizes[gi]
        if not chosen:
            chosen = [int(np.argmin(sizes))]
        test[rows[np.isin(inv, chosen)]] = True
    return test


out = []
print(f"{'dataset':13s} {'seed':>5s} {'split':7s} {'test rows':>10s} {'with identical training row':>30s}")
for ds in a.datasets:
    key, lab, excluded = load(ds)
    n_rows, n_distinct = len(key), len(np.unique(key))
    print(f"{ds}: {n_rows:,} rows, {n_distinct:,} distinct feature vectors "
          f"({100 * (1 - n_distinct / n_rows):.1f}% of rows duplicate another row); excluded classes: {excluded or 'none'}")
    for s in a.seeds:
        for mode in ("random", "group"):
            te = split(key, lab, s, mode)
            twin = np.isin(key[te], key[~te])
            per_class = {c: round(100 * twin[lab[te] == c].mean(), 1) for c in np.unique(lab[te])}
            rec = {"dataset": ds, "seed": s, "split": mode, "test_rows": int(te.sum()),
                   "test_rows_with_train_twin": int(twin.sum()), "pct": round(100 * twin.mean(), 2),
                   "pct_per_class": per_class, "rows": n_rows, "distinct_vectors": n_distinct}
            out.append(rec)
            print(f"{ds:13s} {s:5d} {mode:7s} {rec['test_rows']:10,d} {rec['test_rows_with_train_twin']:>14,d} ({rec['pct']:5.1f}%)")
json.dump(out, open("leak_stats.json", "w"), indent=1)
print("\nSaved leak_stats.json -- please send this file.")

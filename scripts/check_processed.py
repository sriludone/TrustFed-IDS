import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trustfed.data import PROCESSED_DIR
files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_meta.json")))
if not files:
    sys.exit(f"no meta files in {PROCESSED_DIR}; run scripts/preprocess.py first")
for f in files:
    m = json.load(open(f)); c = m["class_counts_final"]
    print(f"\n=== {m['dataset']}: {sum(c.values()):,} rows, {len(c)} classes, {m['n_features']} features")
    print(f"  dedup: {m['dedup']}")
    print(f"  dropped (explicit): {m['dropped_explicit']}")
    print(f"  dropped (pattern):  {m['dropped_by_pattern']}")
    print(f"  dropped (non-numeric): {m['dropped_non_numeric']}")
    if m.get("onehot_columns"):
        print(f"  one-hot: {m['onehot_columns']}")
    print("  classes (final rows):")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        flag = "   <-- small" if v < 200 else ""
        print(f"    {k:32s} {v:>9,}{flag}")

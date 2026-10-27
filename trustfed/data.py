"""
data.py -- preprocessing (raw -> processed) and the ONE shared federated split.

Fixes relative to the original data_loader/training/baseline code:
  * Identifier/leakage columns (IPs, ports, timestamps, flow IDs, MACs) are
    dropped by name pattern; the dropped list is saved in <ds>_meta.json.
  * Exact duplicates are removed, and feature-identical rows with CONFLICTING
    labels are removed, BEFORE any split (duplicates across train/test inflate
    scores, especially on CICIDS2017 and CICIoT2023).
  * Native feature width per dataset (no padding to 76 -- each dataset is its
    own federation, so a common width is unnecessary).
  * Every method (TrustFed, all FL baselines, centralised, LSTM-IDS) receives
    exactly the same test set, the same client partition and the same
    per-client train/val split from prepare_federation(). The original code
    built a different test set in training.py than in the baselines.
  * A per-client validation split is used for model selection / early stopping
    (federated validation). The test set is never used for selection.
  * A small "root" set (FLTrust's server data) is ALWAYS carved out of the
    training pool, even for methods that ignore it, so every method trains on
    identical client data.
"""
import os, re, json
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from .severity import severity_map, benign_index

ROOT = os.environ.get("TRUSTFED_HOME",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

RARE_THRESHOLDS = {"CICIDS2017": 50, "TON_IoT": 50, "CICIoT2023": 200, "Edge-IIoTset": 100}
CAP = {"CICIDS2017": 2_000_000, "TON_IoT": 200_000, "CICIoT2023": 300_000,
       "Edge-IIoTset": 250_000}

# ---------------------------------------------------------------- identifiers
_ID_EXACT = {"ts", "time", "stime", "ltime", "timestamp", "frame.time", "flow id", "flow_id",
             "flowid", "src_ip", "dst_ip", "src_port", "dst_port", "source ip", "destination ip",
             "source port", "destination port", "sport", "dsport", "srcip", "dstip",
             "tcp.srcport", "tcp.dstport", "udp.port", "tcp.port", "eth.src", "eth.dst",
             "ip.src_host", "ip.dst_host", "arp.src.hw.mac", "arp.dst.hw.mac",
             "arp.src.proto_ipv4", "arp.dst.proto_ipv4", "dataset"}
_ID_REGEX = [re.compile(p) for p in (
    r"timestamp", r"(^|[ _.])(src|dst|source|destination)[ _.]?(ip|port|host|addr|mac)",
    r"(^|[ _.])mac($|[ _.])", r"\bhw\.mac\b", r"flow[ _.]?id")]


def identifier_columns(columns, keep=()):
    out = []
    for c in columns:
        n = str(c).strip().lower()
        if c in keep or n == "label":
            continue
        if n in _ID_EXACT or any(r.search(n) for r in _ID_REGEX):
            out.append(c)
    return out


# ---------------------------------------------------------------- preprocessing
def deduplicate(df, feat_cols, mode="keep"):
    """Conflicts: rows with identical features but different labels are resolved by
    MAJORITY (rows of the most frequent label for that feature vector are kept). The
    old rule deleted every row of such a group, which erased whole classes (e.g.
    Edge-IIoTset Fingerprinting).
    mode='keep' : duplicates retained; leakage is prevented later by the group-aware
                  split (identical feature vectors never span train and test).
    mode='drop' : exact duplicates removed as well (robustness check)."""
    n0 = len(df)
    key = pd.util.hash_pandas_object(df[feat_cols], index=False).values
    tmp = pd.DataFrame({"k": key, "y": df["label"].values})
    cnt = tmp.groupby(["k", "y"]).size().reset_index(name="n")
    n_lab = cnt.groupby("k")["y"].transform("size")
    maj = cnt.sort_values("n", ascending=False).drop_duplicates("k").set_index("k")["y"]
    keep = tmp["y"].values == maj.reindex(tmp["k"].values).values
    conflict_keys = set(cnt.loc[n_lab > 1, "k"])
    n_conf_groups = len(conflict_keys)
    df = df.loc[keep]
    n_minority = int((~keep).sum())
    n_exact = 0
    if mode == "drop":
        before = len(df)
        df = df.drop_duplicates(subset=feat_cols + ["label"])
        n_exact = before - len(df)
    k2 = pd.util.hash_pandas_object(df[feat_cols], index=False)
    return df, {"mode": mode, "rows_before": int(n0),
                "conflicting_feature_vectors": int(n_conf_groups),
                "minority_label_rows_removed": n_minority,
                "exact_duplicates_removed": int(n_exact),
                "unique_feature_vectors": int(k2.nunique()), "rows_after": int(len(df))}


def _cap_with_floor(raw, cap, floor, seed=42):
    if len(raw) <= cap:
        return raw
    counts = raw["label"].value_counts()
    protected = counts[counts <= floor].index
    large = counts[counts > floor].index
    budget = max(cap - int(counts[protected].sum()), 0)
    tot = int(counts[large].sum())
    parts = [raw[raw["label"] == c] for c in protected]
    for c in large:
        g = raw[raw["label"] == c]
        parts.append(g.sample(min(len(g), max(1, int(budget * counts[c] / tot))), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def preprocess_dataset(ds, max_rows_per_file=None, keep_columns=()):
    """raw files -> data/processed/<ds>_processed.csv + <ds>_meta.json"""
    from . import raw_loaders as rl
    from .raw_loaders import merge_rare_classes
    loaders = {"CICIDS2017": rl.load_cicids2017, "TON_IoT": rl.load_ton_iot,
               "CICIoT2023": rl.load_ciciot2023, "Edge-IIoTset": rl.load_edge_iiotset}
    raw = loaders[ds](max_rows_per_file=max_rows_per_file)
    thr = RARE_THRESHOLDS.get(ds, 100)
    raw = merge_rare_classes(raw, min_samples=thr)

    dropped = identifier_columns(raw.columns, keep=keep_columns)
    raw = raw.drop(columns=dropped)
    feats = [c for c in raw.columns if c != "label"]
    num = raw[feats].apply(pd.to_numeric, errors="coerce")
    non_numeric = [c for c in feats if num[c].notna().mean() < 0.5]
    num = num.drop(columns=non_numeric).astype("float32")
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = num.assign(label=raw["label"].astype(str).str.strip().values)
    feat_cols = [c for c in df.columns if c != "label"]

    df, dedup = deduplicate(df, feat_cols)
    df = _cap_with_floor(df, CAP.get(ds, 1_000_000), 3 * thr)
    df = merge_rare_classes(df, min_samples=thr)

    vt = VarianceThreshold(1e-5).fit(df[feat_cols].values)  # label-free filter
    kept = [c for c, k in zip(feat_cols, vt.get_support()) if k]
    out = df[kept].copy()
    out.columns = [f"f{i}" for i in range(len(kept))]
    out["label"] = df["label"].values

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out.to_csv(os.path.join(PROCESSED_DIR, f"{ds}_processed.csv"), index=False)
    meta = {"dataset": ds, "n_features": len(kept), "feature_names": kept,
            "dropped_identifier_columns": dropped, "dropped_non_numeric_columns": non_numeric,
            "dropped_low_variance": [c for c in feat_cols if c not in kept],
            "dedup": dedup, "rare_threshold": thr,
            "class_counts": out["label"].value_counts().to_dict()}
    with open(os.path.join(PROCESSED_DIR, f"{ds}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{ds}] dropped identifiers: {dropped}")
    print(f"[{ds}] dedup: {dedup}")
    print(f"[{ds}] features={len(kept)} rows={len(out):,} classes={out['label'].nunique()}")
    return meta


# ---------------------------------------------------------------- shared split
def dirichlet_partition(y, n_clients, alpha, rng, min_size=0, max_tries=1000):
    """Label-skewed Dirichlet partition. Redraws until every client holds at least
    `min_size` rows (the NIID-Bench convention). Tiny clients carry almost no signal
    and, under DP, need very large noise, so a floor makes the federation realistic
    without changing the label skew that alpha controls."""
    for _ in range(max_tries):
        idx = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            ic = np.where(y == c)[0]
            rng.shuffle(ic)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(ic)).astype(int)[:-1]
            for k, part in enumerate(np.split(ic, cuts)):
                idx[k].extend(part.tolist())
        if min(len(i) for i in idx) >= min_size:
            break
    else:
        print(f"  [warn] no partition with min client size {min_size} after {max_tries} draws")
    return [np.array(sorted(i), dtype=int) for i in idx]


def prepare_federation(dataset, n_clients=20, alpha=0.5, seed=42, test_size=0.15,
                       val_frac=0.15, root_per_class=100, processed_dir=None, scaler="quantile",
                       min_client_size=0, min_test_per_class=30, min_unique_patterns=10, split="group"):
    """dataset 'CICIoT2023-8' = CICIoT2023 relabelled to its 8 official families.
    scaler: 'quantile' (default; robust to the heavy tails of flow features, fitted on
    the training pool only, output in [0,1]) or 'minmax'."""
    from .datasets import FAMILY_VIEWS
    base, fam = dataset, None
    for suffix, (b, mapping) in FAMILY_VIEWS.items():
        if dataset == suffix:
            base, fam = b, mapping
    path = os.path.join(processed_dir or PROCESSED_DIR, f"{base}_processed.csv")
    df = pd.read_csv(path, low_memory=False)
    if fam:
        df["label"] = df["label"].map(lambda s: fam.get(s, s))
    feat = [c for c in df.columns if c.startswith("f")]
    # A class needs several distinct feature vectors to be split into unseen-pattern
    # train/test parts (e.g. Edge-IIoTset Fingerprinting has ONE distinct packet after
    # identifier removal). Such classes are excluded from this view and reported.
    _k = pd.util.hash_pandas_object(df[feat], index=False)
    n_pat = pd.Series(_k.values).groupby(df["label"].astype(str).values).nunique()
    dropped_classes = {c: int(n) for c, n in n_pat.items() if n < min_unique_patterns}
    if dropped_classes:
        print(f"  [{dataset}] classes with < {min_unique_patterns} distinct feature vectors excluded: {dropped_classes}")
        df = df[~df["label"].astype(str).isin(dropped_classes)].reset_index(drop=True)
    class_names = sorted(df["label"].astype(str).unique().tolist())
    cmap = {c: i for i, c in enumerate(class_names)}
    y = df["label"].astype(str).map(cmap).values.astype(np.int64)
    X = df[feat].values.astype(np.float32)
    rng = np.random.default_rng(seed)

    idx = np.arange(len(y))
    counts = np.bincount(y, minlength=len(class_names))
    # Group-aware, class-stratified test split: rows with an identical feature vector
    # form one group and go to train OR test, never both (no train/test leakage even
    # though duplicates are kept). With fully de-duplicated data every group is one row.
    grp = pd.util.hash_pandas_object(df[feat], index=False).values
    test_mask = np.zeros(len(y), bool)
    for c in range(len(class_names)):
        rows = np.where(y == c)[0]
        if len(rows) < 2:
            continue
        if split == "random":
            # Comparison protocol only: ordinary class-stratified random split of rows,
            # so identical feature vectors may fall on both sides (reported as `leak`).
            target = int(round(max(test_size * len(rows), min(min_test_per_class, len(rows) // 3))))
            test_mask[rng.permutation(rows)[:max(1, min(target, len(rows) - 1))]] = True
            continue
        g, inv, sizes = np.unique(grp[rows], return_inverse=True, return_counts=True)
        if len(g) < 2:
            continue
        target = max(test_size * len(rows), min(min_test_per_class, len(rows) // 3))
        limit = max(1.5 * target, target + min_test_per_class)   # avoid one huge pattern overshooting
        order = rng.permutation(len(g)); taken, chosen = 0, []
        for gi in order:
            if taken >= target or len(chosen) >= len(g) - 1:
                break
            if taken + sizes[gi] > limit:
                continue
            chosen.append(gi); taken += sizes[gi]
        if not chosen:                                            # every pattern is too big: take the smallest
            chosen = [int(np.argmin(sizes))]
        test_mask[rows[np.isin(inv, chosen)]] = True
    test, pool = idx[test_mask], idx[~test_mask]
    leak = len(set(grp[test]) & set(grp[pool]))
    if scaler == "quantile":
        sc = QuantileTransformer(n_quantiles=1000, output_distribution="uniform",
                                 subsample=200_000, random_state=seed).fit(X[pool])
    else:
        sc = MinMaxScaler().fit(X[pool])
    Xs = np.clip(sc.transform(X), 0, 1).astype(np.float32)

    root = []
    for c in range(len(class_names)):
        ic = pool[y[pool] == c]
        root.extend(rng.permutation(ic)[:min(root_per_class, len(ic) // 4)].tolist())
    root = np.array(sorted(root), dtype=int)
    pool = np.setdiff1d(pool, root)

    parts = dirichlet_partition(y[pool], n_clients, alpha, rng, min_size=min_client_size)
    clients = []
    for k, p in enumerate(parts):
        ids = rng.permutation(pool[p])
        nv = int(round(val_frac * len(ids))) if len(ids) >= 10 else 0
        va, tr = ids[:nv], ids[nv:]
        clients.append({"id": k, "X_train": Xs[tr], "y_train": y[tr],
                        "X_val": Xs[va], "y_val": y[va]})
    sev = severity_map(class_names)
    return {"dataset": dataset, "class_names": class_names, "n_classes": len(class_names),
            "n_features": len(feat), "benign_idx": benign_index(class_names), "severity": sev,
            "X_test": Xs[test], "y_test": y[test], "X_root": Xs[root], "y_root": y[root],
            "clients": clients, "seed": seed, "alpha": alpha,
            "test_class_counts": np.bincount(y[test], minlength=len(class_names)).tolist(),
            "train_test_shared_feature_vectors": int(leak), "split": split,
            "test_rows_with_train_twin": int(np.isin(grp[test], grp[pool]).sum()),
            "dropped_classes": dropped_classes,
            "test_fraction_per_class": {class_names[c]: round(float((y[test] == c).sum() / max((y == c).sum(), 1)), 3)
                                        for c in range(len(class_names))},
            "client_sizes": [len(c["y_train"]) for c in clients]}


# ---------------------------------------------------------------- synthetic
def make_synthetic(name="SYNTH", n=24000, n_features=20, seed=0, processed_dir=None):
    """Imbalanced multi-class stand-in for smoke tests (NOT for reporting)."""
    from sklearn.datasets import make_classification
    names = ["BENIGN", "DDoS", "PortScan", "Backdoor", "Recon-Sweep", "Ransomware"]
    w = [0.55, 0.2, 0.12, 0.07, 0.04, 0.02]
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=12,
                               n_redundant=4, n_classes=len(names), weights=w,
                               class_sep=1.6, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = [names[i] for i in y]
    d = processed_dir or PROCESSED_DIR
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, f"{name}_processed.csv"), index=False)
    return os.path.join(d, f"{name}_processed.csv")

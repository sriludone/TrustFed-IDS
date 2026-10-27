import os, glob, json, warnings, re
import numpy as np
import pandas as pd
from sklearn.preprocessing    import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from collections import defaultdict

warnings.filterwarnings("ignore")


BASE_DIR      = os.environ.get(
    "TRUSTFED_HOME",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PARTITION_DIR = os.path.join(BASE_DIR, "data", "partitions")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
N_FEATURES    = 76

for d in [PROCESSED_DIR, PARTITION_DIR, RESULTS_DIR,
          os.path.join(RESULTS_DIR, "tables"),
          os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(d, exist_ok=True)


def _read_any(filepath, max_rows=None, encoding=None):
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(filepath, low_memory=False, nrows=max_rows,
                             sep=sep, encoding=encoding)
            if df.shape[1] > 3:
                return df
        except Exception:
            continue
    return None


def _find_files(folder):
    seen = {}
    for pat in ["*.csv", "*.pcap_ISCX", "*.tsv", "*.txt", "*.data"]:
        for f in glob.glob(os.path.join(folder, "**", pat), recursive=True):
            seen[f] = 1
        for f in glob.glob(os.path.join(folder, pat)):
            seen[f] = 1
    return list(seen.keys())


def load_cicids2017(max_rows_per_file=None):
    path  = os.path.join(RAW_DIR, "CICIDS2017")
    files = _find_files(path)
    assert files, f"No files in {path}"
    print(f"  [CICIDS2017] {len(files)} files")
    frames = []
    for f in files:
        df = _read_any(f, encoding="cp1252")
        if df is None:
            continue
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        lc = next((c for c in df.columns if c.strip() == "Label"), None)
        if lc is None:
            continue
        df = df.rename(columns={lc: "label"})
        df["label"] = df["label"].astype(str).str.strip()
        if max_rows_per_file and len(df) > max_rows_per_file:
            atk = df[df["label"] != "BENIGN"]
            ben = df[df["label"] == "BENIGN"]
            n   = max(0, max_rows_per_file - len(atk))
            if n < len(ben):
                ben = ben.sample(n, random_state=42)
            df = pd.concat([atk, ben], ignore_index=True)
        frames.append(df)
    assert frames
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = "CICIDS2017"
    print(f"[CICIDS2017] {len(df):,} rows | classes: {df['label'].value_counts().to_dict()}")
    return df


def load_ton_iot(max_rows_per_file=None):
    path  = os.path.join(RAW_DIR, "TON_IoT")
    files = _find_files(path)
    assert files, f"No files in {path}"
    frames = []
    for f in files:
        df = _read_any(f, max_rows=max_rows_per_file)
        if df is None:
            continue
        df.columns = df.columns.str.strip().str.lower()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "type" in df.columns:
            if "label" in df.columns:
                df = df.drop(columns=["label"])
            df = df.rename(columns={"type": "label"})
        elif "label" in df.columns:
            df["label"] = (df["label"].astype(str).str.strip()
                           .map({"0": "TON_Normal", "1": "TON_Attack"})
                           .fillna(df["label"].astype(str).str.strip()))
        elif "attack_type" in df.columns:
            df = df.rename(columns={"attack_type": "label"})
        else:
            continue
        if isinstance(df["label"], pd.DataFrame):
            df["label"] = df["label"].iloc[:, 0]
        df["label"] = df["label"].astype(str).str.strip()
        frames.append(df)
    assert frames
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = "TON_IoT"
    print(f"[TON_IoT] {len(df):,} rows | classes: {df['label'].value_counts().to_dict()}")
    return df


def load_ciciot2023(max_rows_per_file=None):
    path  = os.path.join(RAW_DIR, "CICIoT2023")
    files = _find_files(path)
    assert files, f"No files in {path}"


    effective_cap = max_rows_per_file if max_rows_per_file is not None else 15_000
    print(f"  [CICIoT2023] {len(files)} files ... (cap={effective_cap:,} rows/file "
          f"to stay memory-safe on this ~46.7M-row dataset)")

    LABEL_CANDS = ["label","labels","attack","attack_type","class","category",
                   "target","type","traffic_category","traffic_type","flow_label"]
    STR_DROPS   = {"flow_id","flowid","src_ip","dst_ip","source_ip","destination_ip",
                   "timestamp","time","src_port","dst_port"}
    frames, lcf, vb = [], None, set()
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False, nrows=effective_cap)
        except Exception:
            continue
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        cl = {c.lower(): c for c in df.columns}
        found = None
        if lcf and lcf.lower() in cl:
            found = cl[lcf.lower()]
        else:
            for c in LABEL_CANDS:
                if c in cl:
                    found = cl[c]; lcf = found; break
        if found:
            if found != "label":
                df = df.rename(columns={found: "label"})
            if isinstance(df["label"], pd.DataFrame):
                df["label"] = df["label"].iloc[:, 0]
            df["label"] = df["label"].astype(str).str.strip()
        else:
            base  = os.path.basename(f).replace(".pcap.csv","").replace(".csv","")
            label = re.sub(r'\d+$','',base).rstrip('_').rstrip('-')
            df["label"] = label
            vb.add(label)
        num = df.drop(columns=["label"], errors="ignore")
        for sc in STR_DROPS:
            orig = cl.get(sc)
            if orig and orig in num.columns:
                num = num.drop(columns=[orig])
        num = num.select_dtypes(include=[np.number])
        num["label"] = df["label"].values
        frames.append(num)
    assert frames
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = "CICIoT2023"
    if vb:
        print(f"  [CICIoT2023] Filename-as-label: {len(vb)} classes")
    print(f"[CICIoT2023] {len(df):,} rows | classes: {df['label'].nunique()}")
    return df


def load_edge_iiotset(max_rows_per_file=None):
    path = os.path.join(RAW_DIR, "Edge-IIoTset-dataset")
    DROP = {"frame.time","ip.src_host","ip.dst_host","arp.dst.proto_ipv4",
            "arp.src.proto_ipv4","arp.dst.hw.mac","arp.src.hw.mac",
            "eth.dst","eth.src","eth.type"}


    canonical_dir = os.path.join(path, "Selected dataset for ML and DL")
    canonical_candidates = [
        os.path.join(canonical_dir, "DNN-EdgeIIoT-dataset.csv"),
        os.path.join(canonical_dir, "ML-EdgeIIoT-dataset.csv"),
    ]
    canonical = next((c for c in canonical_candidates if os.path.exists(c)), None)

    if canonical:
        print(f"  [Edge-IIoTset] Using canonical ML-ready file: "
              f"{os.path.basename(canonical)}")
        files = [canonical]
    else:
        files = _find_files(path)
        assert files, f"No files in {path}"
        print(f"  [Edge-IIoTset] WARNING: canonical file not found under "
              f"'Selected dataset for ML and DL/'. Falling back to recursive "
              f"search of {len(files)} files — this risks mixing incompatible "
              f"raw attack/sensor CSVs with different schemas.")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False, nrows=max_rows_per_file)
        except Exception:
            continue
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        if "Attack_type" in df.columns:
            df = df.rename(columns={"Attack_type": "label"})
            if "Attack_label" in df.columns:
                df = df.drop(columns=["Attack_label"])
        elif "Attack_label" in df.columns:
            df["label"] = df["Attack_label"].astype(str).map({"0":"Normal","1":"Attack"}).fillna("Unknown")
            df = df.drop(columns=["Attack_label"])
        else:
            df["label"] = "Normal"
        df = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")


        for c in list(df.columns):
            if c in ("label", "dataset"):
                continue


            if not pd.api.types.is_numeric_dtype(df[c]):
                coerced = pd.to_numeric(df[c], errors="coerce")


                if coerced.notna().mean() >= 0.5:
                    df[c] = coerced
                else:
                    df = df.drop(columns=[c])
        df["label"] = df["label"].astype(str).str.strip()
        frames.append(df)
    assert frames
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = "Edge-IIoTset"
    print(f"[Edge-IIoTset] {len(df):,} rows | "
          f"feature cols: {len([c for c in df.columns if c not in ('label','dataset')])} | "
          f"classes: {sorted(df['label'].unique())}")
    return df


def merge_rare_classes(df: pd.DataFrame, min_samples: int = 100,
                       merged_name: str = "Rare_Attack") -> pd.DataFrame:
    counts = df["label"].value_counts()
    rare   = counts[counts < min_samples].index.tolist()
    if rare:
        print(f"  Merging {len(rare)} rare classes (<{min_samples} samples) -> '{merged_name}'")
        print(f"  Merged: {rare}")
        df["label"] = df["label"].apply(lambda x: merged_name if x in rare else x)
    return df


_FEAT_CACHE = {}


def align_features(df: pd.DataFrame, ds_name: str,
                   n_features: int = N_FEATURES) -> pd.DataFrame:
    label  = df["label"].copy()
    ds_col = df["dataset"].copy()
    num    = df.drop(columns=["label","dataset"], errors="ignore")
    num.columns = num.columns.str.strip()
    num    = num.apply(pd.to_numeric, errors="coerce").astype("float32")
    num.replace([np.inf, -np.inf], np.nan, inplace=True)
    num.fillna(0.0, inplace=True)

    if ds_name not in _FEAT_CACHE:
        sel = VarianceThreshold(threshold=1e-5)
        try:
            sel.fit(num.values[:min(50000, len(num))])
            _FEAT_CACHE[ds_name] = sel.get_support()
        except Exception:
            _FEAT_CACHE[ds_name] = np.ones(num.shape[1], dtype=bool)

    mask = _FEAT_CACHE[ds_name]
    if len(mask) == num.shape[1]:
        num = num.loc[:, mask]

    cols = num.columns.tolist()
    if len(cols) >= n_features:
        num = num.iloc[:, :n_features]
    else:
        for i in range(len(cols), n_features):
            num[f"pad_{i}"] = 0.0

    num.columns    = [f"f{i}" for i in range(n_features)]
    num["label"]   = label.values
    num["dataset"] = ds_col.values
    return num


def encode_labels(df, le=None):
    if le is None:
        le = LabelEncoder()
        le.fit(df["label"].astype(str))
    known = set(le.classes_)
    df["label"] = df["label"].astype(str).apply(lambda x: x if x in known else "unknown")
    df["label_enc"] = le.transform(df["label"]).astype("int64")
    label_map = {int(i): str(c) for i, c in enumerate(le.classes_)}
    return df, le, label_map


def partition_non_iid(df, n_clients=20, alpha=0.5, seed=42):
    np.random.seed(seed)
    client_idx = defaultdict(list)
    for c in df["label_enc"].unique():
        idx_c = np.where(df["label_enc"].values == c)[0]
        np.random.shuffle(idx_c)
        props  = np.random.dirichlet(np.repeat(alpha, n_clients))
        splits = (props * len(idx_c)).astype(int)
        splits[-1] = len(idx_c) - splits[:-1].sum()
        ptr = 0
        for cid, n in enumerate(splits):
            client_idx[cid].extend(idx_c[ptr:ptr+n].tolist())
            ptr += n

    partition = {}
    for cid in range(n_clients):
        idx = np.array(client_idx.get(cid, []))
        np.random.shuffle(idx)
        n   = len(idx)
        if n == 0:
            partition[cid] = {"train": [], "val": [], "test": []}
            continue
        n_tr = max(1, int(0.70 * n))
        n_va = max(1, int(0.15 * n))
        n_te = max(1, n - n_tr - n_va)
        if n_tr + n_va + n_te > n:
            n_tr = n; n_va = 0; n_te = 0
        partition[cid] = {
            "train": idx[:n_tr].tolist(),
            "val":   idx[n_tr:n_tr+n_va].tolist(),
            "test":  idx[n_tr+n_va:n_tr+n_va+n_te].tolist(),
        }
    return partition


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    w = np.ones(n_classes, dtype="float32")
    if len(y) == 0:
        return w
    classes, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    for c, cnt in zip(classes, counts):
        if c < n_classes:
            w[c] = np.sqrt(total / (n_classes * max(cnt, 1)))
    w /= w.mean()
    return np.clip(w, 0.1, 20.0)


def get_client_arrays(df, partition, client_id):
    feat_cols = [f"f{i}" for i in range(N_FEATURES)]
    df = df.reset_index(drop=True)

    def safe(idx_list):
        idx = [i for i in idx_list if 0 <= i < len(df)]
        if not idx:
            return np.zeros((0, N_FEATURES), dtype="float32"), np.zeros((0,), dtype="int64")
        return (df.iloc[idx][feat_cols].values.astype("float32"),
                df.iloc[idx]["label_enc"].values.astype("int64"))

    return safe(partition[client_id]["train"]), \
           safe(partition[client_id]["val"]), \
           safe(partition[client_id]["test"])


def save_processed(df, name):
    out = os.path.join(PROCESSED_DIR, f"{name}_processed.csv")
    df.to_csv(out, index=False)
    print(f"  Saved -> {out}  ({len(df):,} rows)")


def save_partition_json(partition, name, label_map):
    d = os.path.join(PARTITION_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "partition.json"), "w") as f:
        json.dump({str(k): v for k, v in partition.items()}, f, indent=2)
    with open(os.path.join(d, "label_map.json"), "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)


def load_partition_json(name):
    d = os.path.join(PARTITION_DIR, name)
    with open(os.path.join(d, "partition.json")) as f:
        partition = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(d, "label_map.json")) as f:
        label_map = json.load(f)
    return partition, label_map


def run_pipeline(n_clients=20, alpha=0.5, max_rows=None, datasets=None):
    if datasets is None:
        datasets = ["CICIDS2017", "TON_IoT", "CICIoT2023", "Edge-IIoTset"]

    loaders = {
        "CICIDS2017":   load_cicids2017,
        "TON_IoT":      load_ton_iot,
        "CICIoT2023":   load_ciciot2023,
        "Edge-IIoTset": load_edge_iiotset,
    }


    RARE_THRESHOLDS = {
        "CICIDS2017":  50,
        "TON_IoT":     50,
        "CICIoT2023":  200,
        "Edge-IIoTset": 100,
    }

    CAP = {
        "CICIDS2017":  2_000_000,
        "TON_IoT":       200_000,
        "CICIoT2023":    300_000,
        "Edge-IIoTset":  250_000,
    }

    print("\n[Pass 1] Loading all datasets ...")
    all_labels = set()
    raw_dfs    = {}
    for ds in datasets:
        print(f"  Loading {ds} ...")
        raw = loaders[ds](max_rows_per_file=max_rows)

        raw = merge_rare_classes(raw, min_samples=RARE_THRESHOLDS.get(ds, 100))
        raw_dfs[ds] = raw
        all_labels.update(raw["label"].astype(str).unique())

    print(f"\n[Pass 1] Unique labels ({len(all_labels)}): {sorted(all_labels)}")


    all_labels.add("Rare_Attack")
    all_labels.add("unknown")

    le = LabelEncoder()
    le.fit(sorted(all_labels))
    print(f"[Pass 1] LabelEncoder: {len(le.classes_)} classes "
          f"(incl. reserved Rare_Attack / unknown)\n")

    results   = {}
    label_map = None
    for ds in datasets:
        print(f"\n{'='*60}\nProcessing {ds}\n{'='*60}")
        raw = raw_dfs[ds]
        cap = CAP.get(ds, 1_000_000)
        if len(raw) > cap:
            print(f"  Sampling {cap:,} from {len(raw):,} rows ...")


            floor = 3 * RARE_THRESHOLDS.get(ds, 100)
            try:
                counts_now = raw["label"].value_counts()
                protected  = counts_now[counts_now <= floor].index.tolist()
                large      = counts_now[counts_now >  floor].index.tolist()
                reserved_n = int(counts_now.loc[protected].sum()) if protected else 0
                budget     = max(cap - reserved_n, 0)
                large_total= int(counts_now.loc[large].sum()) if large else 0
                parts = []
                for c in protected:
                    parts.append(raw[raw["label"] == c])
                for c in large:
                    n_target = max(1, int(budget * counts_now[c] / max(large_total, 1)))
                    grp = raw[raw["label"] == c]
                    n_target = min(n_target, len(grp))
                    parts.append(grp.sample(n_target, random_state=42))
                raw = pd.concat(parts, ignore_index=True).reset_index(drop=True)
                if protected:
                    print(f"  Floor-protected {len(protected)} small class(es) "
                          f"(<= {floor} samples): kept in full -> {protected}")
            except Exception:
                raw = raw.sample(n=min(cap, len(raw)), random_state=42).reset_index(drop=True)


        raw = merge_rare_classes(raw, min_samples=RARE_THRESHOLDS.get(ds, 100))

        df = align_features(raw, ds, n_features=N_FEATURES)
        df, le, label_map = encode_labels(df, le)
        df = df.reset_index(drop=True)
        save_processed(df, ds)
        partition = partition_non_iid(df, n_clients=n_clients, alpha=alpha)
        save_partition_json(partition, ds, label_map)
        results[ds] = (df, partition, label_map)
        vc = df["label"].value_counts().to_dict()
        print(f"  Classes: {df['label'].nunique()} | min_count: {min(vc.values())} | "
              f"Clients: {n_clients}")
        print(f"  Distribution: {vc}")

    glm = os.path.join(PARTITION_DIR, "global_label_map.json")
    with open(glm, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"\nGlobal label map -> {glm} ({len(label_map)} classes)")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",   action="store_true")
    parser.add_argument("--dataset", default=None)
    args     = parser.parse_args()
    max_rows = 10000 if args.quick else None
    datasets = [args.dataset] if args.dataset else None
    results  = run_pipeline(n_clients=20, alpha=0.5,
                            max_rows=max_rows, datasets=datasets)
    print()
    for ds, (df, part, lmap) in results.items():
        vc = df["label_enc"].value_counts()
        print(f"{ds}: rows={len(df):,}  classes={df['label'].nunique()}  "
              f"min={vc.min()}  max={vc.max()}  client0_train={len(part[0]['train'])}")
    print("data_loader.py OK")

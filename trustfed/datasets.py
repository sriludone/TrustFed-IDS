import os, re, glob, json
import numpy as np
import pandas as pd

from .data import identifier_columns, deduplicate, PROCESSED_DIR, ROOT

RAW_DIR = os.path.join(ROOT, "data", "raw")


FAMILY_CICIOT = {
    "BenignTraffic": "Benign",
    **{k: "DDoS" for k in ["DDoS-RSTFINFlood", "DDoS-PSHACK_Flood", "DDoS-SYN_Flood", "DDoS-UDP_Flood",
                           "DDoS-TCP_Flood", "DDoS-ICMP_Flood", "DDoS-SynonymousIP_Flood",
                           "DDoS-ACK_Fragmentation", "DDoS-UDP_Fragmentation",
                           "DDoS-ICMP_Fragmentation", "DDoS-SlowLoris", "DDoS-HTTP_Flood"]},
    **{k: "DoS" for k in ["DoS-UDP_Flood", "DoS-SYN_Flood", "DoS-TCP_Flood", "DoS-HTTP_Flood"]},
    **{k: "Mirai" for k in ["Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain"]},
    **{k: "Recon" for k in ["Recon-PingSweep", "Recon-OSScan", "Recon-PortScan",
                            "VulnerabilityScan", "Recon-HostDiscovery"]},
    **{k: "Spoofing" for k in ["DNS_Spoofing", "MITM-ArpSpoofing"]},
    **{k: "Web" for k in ["BrowserHijacking", "Backdoor_Malware", "XSS", "Uploading_Attack",
                          "SqlInjection", "CommandInjection"]},
    "DictionaryBruteForce": "BruteForce",
}
CICIDS_RARE = {"Heartbleed", "Infiltration", "Web Attack - Sql Injection"}


FAMILY_EDGE = {"Normal": "Normal",
               **{k: "DoS_DDoS" for k in ["DDoS_UDP", "DDoS_ICMP", "DDoS_TCP", "DDoS_HTTP"]},
               **{k: "Information_gathering" for k in ["Port_Scanning", "Vulnerability_scanner", "Fingerprinting"]},
               "MITM": "MITM",
               **{k: "Injection" for k in ["SQL_injection", "XSS", "Uploading"]},
               **{k: "Malware" for k in ["Backdoor", "Password", "Ransomware"]}}


FAMILY_VIEWS = {"CICIoT2023-8": ("CICIoT2023", FAMILY_CICIOT),
                "Edge-IIoTset-6": ("Edge-IIoTset", FAMILY_EDGE)}


SETTINGS = {
    "Edge-IIoTset": {
        "label_col": "Attack_type", "also_drop": ["Attack_label"],


        "drop": ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
                 "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
                 "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
                 "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
                 "dns.qry.name", "udp.stream", "tcp.ack_raw", "tcp.checksum", "icmp.checksum",
                 "mbtcp.trans_id"],
        "onehot": ["http.request.method", "http.referer", "http.request.version",
                   "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic"],
        "cap_per_class": 400_000,
    },
    "CICIoT2023": {
        "label_col": "label", "drop": [], "onehot": [],
        "two_pass": True,
        "cap_per_class": 60_000,
    },
    "CICIDS2017": {
        "label_col": "Label", "drop": ["Fwd Header Length.1"], "onehot": [],
        "cap_per_class": 400_000,
    },
}


def _norm_label(s):
    s = str(s).strip()
    s = s.encode("ascii", "replace").decode().replace("?", "-")
    s = re.sub(r"\s*-+\s*", " - ", s) if s.startswith("Web Attack") else s
    return re.sub(r"\s+", " ", s)


def _shrink(df):
    f = df.select_dtypes("float64").columns
    if len(f):
        df[f] = df[f].astype(np.float32)
    return df


def _read_files(paths, label_col, rows_per_file=None, seed=0, encoding=None, skip=()):
    rng = np.random.default_rng(seed); frames = []
    skip = {s.strip() for s in skip}
    for i, p in enumerate(sorted(paths)):
        df = pd.read_csv(p, low_memory=False, encoding=encoding, encoding_errors="replace",
                         usecols=lambda c: c.strip() not in skip)
        df = _shrink(df)
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        if label_col not in df.columns:
            print(f"  [skip] {os.path.basename(p)}: no '{label_col}' column"); continue
        if rows_per_file and len(df) > rows_per_file:
            df = df.iloc[rng.choice(len(df), rows_per_file, replace=False)]
        frames.append(df)
        print(f"  read {os.path.basename(p)}: {len(df):,} rows")
    if not frames:
        raise FileNotFoundError("no usable CSV files found")
    return pd.concat(frames, ignore_index=True)


def _canon(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_CICIOT_CANON = {_canon(k): k for k in FAMILY_CICIOT}
_CICIOT_CANON.update({_canon("Benign_Final"): "BenignTraffic", _canon("Benign"): "BenignTraffic"})


def folder_label(path):
    for part in reversed(os.path.normpath(path).split(os.sep)[:-1]):
        lab = _CICIOT_CANON.get(_canon(part))
        if lab:
            return lab
    return None


def _file_labels(p, label_col):
    head = pd.read_csv(p, nrows=0)
    if label_col in [c.strip() for c in head.columns]:
        return "column", None
    lab = folder_label(p)
    return ("folder", lab) if lab else (None, None)


def _read_two_pass(paths, label_col, cap, seed=0):
    rng = np.random.default_rng(seed); counts = {}; kind = {}; skipped = []
    modes = set()
    for p in sorted(paths):
        mode, flab = _file_labels(p, label_col)
        if mode is None:
            skipped.append(p); continue
        kind[p] = (mode, flab); modes.add(mode)
        if mode == "column":
            lab = pd.read_csv(p, usecols=lambda c: c.strip() == label_col).iloc[:, 0]
            for k, v in lab.astype(str).str.strip().value_counts().items():
                counts[k] = counts.get(k, 0) + v
        else:
            with open(p, "rb") as fh:
                n = max(sum(1 for _ in fh) - 1, 0)
            counts[flab] = counts.get(flab, 0) + n
    if skipped:
        print(f"  [skip] {len(skipped)} file(s) with no label column and no class folder: "
              f"{[os.path.relpath(s) for s in skipped][:5]}")
    if modes == {"column", "folder"}:
        print("  [warn] both labelled (merged) files and per-class folders found; the same flows may "
              "appear twice. Keep only one release in data/raw/CICIoT2023 (dedup removes exact copies).")
    prob = {k: min(1.0, 1.2 * cap / v) for k, v in counts.items()}
    print(f"  pass 1: {sum(counts.values()):,} rows, {len(counts)} classes, label source: {sorted(modes)}")
    frames = []
    for p in sorted(kind):
        mode, flab = kind[p]
        df = _shrink(pd.read_csv(p, low_memory=False))
        df.columns = df.columns.str.strip()
        if mode == "folder":
            df[label_col] = flab
        lab = df[label_col].astype(str).str.strip()
        keep = rng.random(len(df)) < lab.map(prob).fillna(1.0).values
        frames.append(df.loc[keep])
    return pd.concat(frames, ignore_index=True), counts


ALIASES = {"Edge-IIoTset": ["Edge-IIoTset", "Edge-IIoTset-dataset", "Edge_IIoTset", "EdgeIIoTset"],
           "CICIoT2023": ["CICIoT2023", "CIC-IoT2023", "CIC_IoT2023", "CICIOT2023"],
           "CICIDS2017": ["CICIDS2017", "CIC-IDS2017", "CIC_IDS2017", "CICIDS-2017"]}


def _find(ds, raw_dir):
    names = ALIASES.get(ds, [ds])
    base = next((os.path.join(raw_dir, n) for n in names if os.path.isdir(os.path.join(raw_dir, n))),
                os.path.join(raw_dir, ds))
    print(f"  raw folder: {base}")
    if ds == "Edge-IIoTset":
        c = glob.glob(os.path.join(base, "**", "DNN-EdgeIIoT-dataset.csv"), recursive=True)
        if not c:
            raise FileNotFoundError(f"DNN-EdgeIIoT-dataset.csv not found under {base}")
        return c[:1]
    files = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no CSV files under {base}")
    return files


def cap_per_class(df, cap, seed=0):
    parts = []
    for c, g in df.groupby("label", sort=False):
        parts.append(g.sample(cap, random_state=seed) if len(g) > cap else g)
    return pd.concat(parts, ignore_index=True)


def preprocess(ds, raw_dir=None, cap=None, keep=(), drop_extra=(), seed=0, dedup="keep"):
    st = SETTINGS[ds]; raw_dir = raw_dir or RAW_DIR
    files = _find(ds, raw_dir)
    if st.get("two_pass"):
        df, _ = _read_two_pass(files, st["label_col"], cap or st["cap_per_class"], seed)
    else:
        df = _read_files(files, st["label_col"], None, seed,
                         encoding="cp1252" if ds == "CICIDS2017" else None, skip=st["drop"])
    df = df.rename(columns={st["label_col"]: "label"}).drop(columns=st.get("also_drop", []), errors="ignore")
    df["label"] = df["label"].map(_norm_label)
    raw_counts = df["label"].value_counts().to_dict()

    if ds == "CICIDS2017":
        df["label"] = df["label"].where(~df["label"].isin(CICIDS_RARE), "Rare_Attack")
    if ds == "CICIoT2023":
        unknown = sorted(set(df["label"]) - set(FAMILY_CICIOT))
        if unknown:
            print(f"  [warn] labels not in the official 34-class list: {unknown}")

    header = [c.strip() for c in pd.read_csv(files[0], nrows=0, encoding_errors="replace").columns]
    explicit = [c for c in st["drop"] if c in header]
    df = df.drop(columns=[c for c in explicit if c in df.columns])
    pattern = [c for c in identifier_columns(df.columns, keep=keep) if c not in explicit]
    extra = [c for c in drop_extra if c in df.columns]
    df = df.drop(columns=pattern + extra)

    onehot = [c for c in st["onehot"] if c in df.columns]
    if onehot:
        df[onehot] = df[onehot].astype(str).fillna("NA")
        top = {c: df[c].value_counts().index[:20] for c in onehot}
        for c in onehot:
            df[c] = df[c].where(df[c].isin(top[c]), "other")
        df = pd.get_dummies(df, columns=onehot, prefix=onehot, dtype=np.float32)

    feats = [c for c in df.columns if c != "label"]
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    non_numeric = [c for c in feats if num[c].notna().mean() < 0.5]
    num = num.drop(columns=non_numeric).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    df = num.assign(label=df["label"].values)
    feats = [c for c in df.columns if c != "label"]

    df, dedup = deduplicate(df, feats, mode=dedup)
    after_dedup = df["label"].value_counts().to_dict()
    df = cap_per_class(df, cap or st["cap_per_class"], seed)
    keep_cols = [c for c in feats if df[c].nunique() > 1]
    out = df[keep_cols].copy(); out.columns = [f"f{i}" for i in range(len(keep_cols))]
    out["label"] = df["label"].values

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out.to_csv(os.path.join(PROCESSED_DIR, f"{ds}_processed.csv"), index=False)
    meta = {"dataset": ds, "files": [os.path.basename(f) for f in files], "n_features": len(keep_cols),
            "feature_names": keep_cols, "dropped_explicit": explicit, "dropped_by_pattern": pattern,
            "dropped_extra": extra, "dropped_non_numeric": non_numeric, "onehot_columns": onehot,
            "dropped_constant": [c for c in feats if c not in keep_cols], "dedup": dedup,
            "cap_per_class": cap or st["cap_per_class"], "class_counts_raw": raw_counts,
            "class_counts_after_dedup": after_dedup,
            "class_counts_final": out["label"].value_counts().to_dict()}
    json.dump(meta, open(os.path.join(PROCESSED_DIR, f"{ds}_meta.json"), "w"), indent=2)
    print(f"[{ds}] {len(out):,} rows, {out['label'].nunique()} classes, {len(keep_cols)} features")
    print(f"[{ds}] dropped: explicit={explicit} pattern={pattern} non-numeric={non_numeric}")
    print(f"[{ds}] dedup: {dedup}")
    return meta

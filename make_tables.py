"""
make_tables.py -- turns results/ (one JSON per run) into every table and figure.
Outputs in results/tables/: CSV + LaTeX per table, figures as PDF/PNG, and
all_runs.csv (one row per run -- the single file to share).
Statistics: paired over seeds; Wilcoxon signed-rank and paired t-test vs TrustFed,
Holm-corrected within each table; effect = mean difference (percentage points).
"""
import os, sys, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trustfed.io import load_runs

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); a = ap.parse_args()
OUT = os.path.join(a.results, "tables"); os.makedirs(OUT, exist_ok=True)
RUNS = load_runs(a.results)
if not RUNS:
    raise SystemExit(f"no result files found under {a.results}/<group>/*.json")


def row(r):
    t = r["test"]
    return {"group": r["group"], "dataset": r["dataset"], "method": r["method"], "seed": r["seed"],
            "attack": r.get("attack", "none"), "frac": r.get("byz_frac", 0.0), "start": r.get("attack_start", 1),
            "alpha": r.get("alpha"), "eps_setting": r.get("eps_setting"),
            "acc": 100 * t["accuracy"], "balacc": 100 * t["balanced_accuracy"], "f1": 100 * t["f1_macro"],
            "auc": t.get("auc_roc"), "benign_far": 100 * t.get("benign_far", np.nan), "fpr_macro": 100 * t["fpr_macro"],
            "eps_median": r.get("epsilon_median"), "eps_max": r.get("epsilon_max"),
            "detect": r.get("byz_detect_rate_mean"), "false_flag": r.get("honest_false_flag_rate_mean"),
            "best_round": r.get("best_val_round"), "runtime_min": r.get("runtime_sec", 0) / 60,
            "mean_round_sec": r.get("mean_round_sec"), "n_params": r.get("n_params"),
            "shared_vectors": r.get("train_test_shared_feature_vectors"),
            "dropped_classes": ";".join(r.get("dropped_classes", {}) or {}), "file": os.path.basename(r["_file"])}


df = pd.DataFrame([row(r) for r in RUNS])
df.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
print(f"  {len(df)} runs loaded; wrote all_runs.csv")
if df["shared_vectors"].fillna(0).astype(float).max() > 0:
    print("  [WARNING] some runs report train/test shared feature vectors > 0")


def ms(x, d=2):
    x = pd.Series(x).dropna()
    return "--" if x.empty else (f"{x.mean():.{d}f}" if len(x) == 1 else f"{x.mean():.{d}f}$\\pm${x.std(ddof=1):.{d}f}")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj


def paired(d, ref, metric):
    from scipy.stats import wilcoxon, ttest_rel
    out = []
    rr = d[d.method == ref].set_index(["dataset", "seed"])[metric]
    for m in sorted(set(d.method) - {ref}):
        o = d[d.method == m].set_index(["dataset", "seed"])[metric]
        j = rr.index.intersection(o.index)
        if len(j) < 3:
            out.append((m, np.nan, np.nan, np.nan)); continue
        diff = rr[j] - o[j]
        pw = wilcoxon(rr[j], o[j]).pvalue if (diff != 0).any() else 1.0
        out.append((m, diff.mean(), pw, ttest_rel(rr[j], o[j]).pvalue))
    t = pd.DataFrame(out, columns=["method", "diff_pp", "p_wilcoxon", "p_ttest"])
    if len(t):
        t["p_wilcoxon_holm"] = holm(t.p_wilcoxon.fillna(1)); t["p_ttest_holm"] = holm(t.p_ttest.fillna(1))
    return t


def to_tex(d, name, caption):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(f"% {caption}\n" + d.to_latex(index=False, escape=False))
    d.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    print(f"  wrote {name}")


by_file = {os.path.basename(r["_file"]): r for r in RUNS}

# ---- main results + privacy budget
m = df[df.group == "main"]
if len(m):
    t = m.groupby("dataset").apply(lambda g: pd.Series({
        "Accuracy": ms(g.acc), "Balanced acc.": ms(g.balacc), "Macro-F1": ms(g.f1), "AUC-ROC": ms(g.auc, 4),
        "Benign FAR (\\%)": ms(g.benign_far), "Macro FPR (\\%)": ms(g.fpr_macro),
        "$\\varepsilon$ max": ms(g.eps_max), "Seeds": len(g)})).reset_index()
    to_tex(t, "table_main", "Per-dataset results, test set at best validation round")
    for ds, g in m.groupby("dataset"):
        recs = [by_file[f] for f in g.file]
        names = recs[0]["class_names"]
        f1 = np.array([r["test"]["per_class"]["f1"] for r in recs])
        pd.DataFrame({"class": names, "f1_mean": f1.mean(0), "f1_sd": f1.std(0),
                      "test_support": recs[0]["test"]["per_class"]["support"]}).to_csv(
            os.path.join(OUT, f"per_class_{ds}.csv"), index=False)
        pd.DataFrame(recs[0]["test"]["confusion_matrix"], index=names, columns=names).to_csv(
            os.path.join(OUT, f"confusion_{ds}_seed{recs[0]['seed']}.csv"))

# ---- baselines (+ TrustFed from main)
if len(df[df.group == "baselines"]):
    b = pd.concat([df[df.group == "baselines"], m])
    rows = []
    for meth, g in b.groupby("method"):
        rows.append({"Method": meth, **{ds: ms(g[g.dataset == ds].f1) for ds in sorted(b.dataset.unique())},
                     "Mean macro-F1": f"{g.groupby('dataset').f1.mean().mean():.2f}",
                     "Mean bal. acc.": f"{g.groupby('dataset').balacc.mean().mean():.2f}",
                     "Mean benign FAR": f"{g.groupby('dataset').benign_far.mean().mean():.2f}"})
    to_tex(pd.DataFrame(rows), "table_baselines", "Macro-F1 per dataset (mean+-sd over seeds)")
    to_tex(paired(b, "trustfed", "f1").round(4), "stats_baselines_f1", "Paired tests vs TrustFed (macro-F1)")

# ---- privacy
pv = df[df.group == "privacy"]
if len(pv):
    t = pv.groupby(["dataset", "eps_setting"]).apply(lambda g: pd.Series({
        "Macro-F1": ms(g.f1), "Balanced acc.": ms(g.balacc), "Accuracy": ms(g.acc),
        "measured eps (max)": ms(g.eps_max), "Seeds": len(g)})).reset_index()
    to_tex(t, "table_privacy", "Privacy-utility trade-off (target epsilon per client, worst case)")

# ---- Byzantine
z = df[df.group == "byzantine"]
if len(z):
    z = z.assign(setting=z.attack + "@" + (100 * z.frac).round().astype(int).astype(str) + "%"
                 + np.where(z.start > 1, "(adaptive)", ""))
    to_tex(z.groupby(["dataset", "method", "setting"]).f1.apply(ms).unstack().reset_index(),
           "table_byzantine_f1", "Macro-F1 under attack (all methods with DP)")
    to_tex(z.groupby(["dataset", "method", "setting"]).apply(
        lambda g: f"{ms(g.detect)} / {ms(g.false_flag)}").unstack().reset_index(),
        "table_byzantine_detection", "Detection rate / honest false-flag rate")

# ---- ablation
ab = df[df.group == "ablation"]
if len(ab):
    to_tex(ab.groupby(["dataset", "method", "attack"]).f1.apply(ms).unstack().reset_index(),
           "table_ablation", "Ablation (macro-F1)")
    for atk in ab.attack.unique():
        to_tex(paired(ab[ab.attack == atk], "trustfed", "f1").round(4), f"stats_ablation_{atk}", "")

# ---- alpha
al = df[df.group == "alpha"]
if len(al):
    to_tex(al.groupby(["alpha", "method"]).f1.apply(ms).unstack().reset_index(), "table_alpha",
           "Heterogeneity sweep (macro-F1)")

# ---- audit
au = [{"dataset": r["dataset"], "seed": r["seed"], **x} for r in RUNS for x in r.get("audit_tamper_suite", [])]
if au:
    au = pd.DataFrame(au)
    to_tex(au.groupby("scenario").agg(detected=("detected", "mean")).reset_index(), "table_audit",
           "Tamper detection (fraction of runs in which tampering was detected)")
    shap_s = [l["shap_sec"] for r in RUNS if r["group"] == "main" for l in r["rounds"] if l.get("shap_sec")]
    if shap_s:
        print(f"  audit: mean explanation time per client-round {np.mean(shap_s):.3f}s")

# ---- figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if len(m):
        fig, ax = plt.subplots(figsize=(7, 4))
        for ds, g in m.groupby("dataset"):
            curves = [[100 * l["val_balacc"] for l in by_file[f]["rounds"]] for f in g.file]
            L = min(map(len, curves)); C = np.array([c[:L] for c in curves]); x = np.arange(1, L + 1)
            ax.plot(x, C.mean(0), label=ds); ax.fill_between(x, C.mean(0) - C.std(0), C.mean(0) + C.std(0), alpha=0.2)
        ax.set_xlabel("Federated round"); ax.set_ylabel("Validation balanced accuracy (%)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_convergence.pdf")); fig.savefig(os.path.join(OUT, "fig_convergence.png"), dpi=300)
    if len(z):
        for ds, zz in z[(z.attack == "label_flip") & (z.start == 1)].groupby("dataset"):
            fig, ax = plt.subplots(figsize=(7, 4))
            for meth, g in zz.groupby("method"):
                s = g.groupby("frac").f1.agg(["mean", "std"])
                ax.errorbar(100 * s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=meth)
            ax.set_xlabel("Malicious clients (%)"); ax.set_ylabel("Macro-F1 (%)"); ax.set_title(ds); ax.legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_byzantine_{ds}.pdf")); fig.savefig(os.path.join(OUT, f"fig_byzantine_{ds}.png"), dpi=300)
    print("  figures written")
except Exception as e:
    print("  figures skipped:", e)

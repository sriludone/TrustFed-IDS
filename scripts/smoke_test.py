"""End-to-end check on synthetic data (CPU, ~10 min). NOT for reporting.
   python scripts/smoke_test.py                 # all stages
   python scripts/smoke_test.py --stage methods42   # one stage: methods42 methods123
                                                  #   methods256 attacks (or attacks:<rule>) preprocess tables"""
import os, sys, json, shutil, time, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, yaml
from trustfed.data import make_synthetic, prepare_federation
from trustfed.runner import run_fl, PRESETS
from trustfed.centralised import run_centralised
from trustfed.agent import run_agents
from trustfed.io import run_name, run_path, write_json

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--stage", default="all"); STAGE = _ap.parse_args().stage
def run_stage(name):
    return STAGE in ("all", name)
T0 = time.time()
R = os.path.join(ROOT, "results_smoke"); P = os.path.join(ROOT, "data_smoke")
if STAGE in ("all", "methods42"):
    shutil.rmtree(R, ignore_errors=True)
make_synthetic("SYNTH", n=12000, n_features=20, processed_dir=P)
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
cfg.update(n_clients=6, n_rounds=3, local_epochs=1, batch_size=128, hidden=32, n_blocks=1,
           agent_train_episodes=8, agent_test_episodes=3, agent_ep_len=200, centralised_epochs=2,
           early_stop_patience=10)
fed = prepare_federation("SYNTH", 6, 0.5, 42, root_per_class=50, processed_dir=P)
print("classes:", fed["class_names"], "severity:", fed["severity"], "benign:", fed["benign_idx"])
def save(g, m, summ, logs, atk="none", fr=0.0, st=1, s=42, extra=None):
    path = run_path(R, g, run_name("SYNTH", m, atk, fr, st, 0.5, s))
    write_json(path, {"group": g, **summ, "rounds": logs, **(extra or {})})
    return path
fails = []

# 1. every method, two seeds for paired stats
for s in (42, 123, 256):
    if not run_stage(f"methods{s}"):
        continue
    for m in PRESETS:
        g = "main" if m == "trustfed" else ("ablation" if m.startswith("abl_") else "baselines")
        afile = run_path(R, g, run_name("SYNTH", m, seed=s))[:-5] + ".audit.npz" if m == "trustfed" else None
        os.makedirs(os.path.join(R, g), exist_ok=True)
        summ, w, logs = run_fl(fed, cfg, m, s, audit=(m == "trustfed"), audit_file=afile, verbose=False)
        extra = {}
        if summ["use_dp"] and len(summ["epsilon_per_client"]) == 0:
            fails.append(f"{m}: DP on but no epsilon recorded")
        if m == "trustfed" and s == 42:
            av = summ["audit_verify"]
            if not all(v["valid"] for v in av.values()):
                fails.append("audit chain failed verification")
            ts = {r["scenario"]: r["detected"] for r in summ["audit_tamper_suite"]}
            print("tamper suite:", ts)
            exp = {k: (k != "no modification (control)" and "no anchors" not in k) for k in ts}
            if ts != exp:
                fails.append(f"tamper suite unexpected: {ts}")
            ag = run_agents(fed, w, cfg, s, verbose=False)
            extra["agent"] = ag
            print("agent:", {p: round(v["action_accuracy"][0], 3) for p, v in ag["summary_mean_sd"].items()})
        save(g, m, summ, logs, s=s, extra=extra)
    for m, arch in (("centralised", "residual_mlp"), ("lstm_ids", "bilstm")):
        summ, _, logs = run_centralised(fed, cfg, s, arch=arch, verbose=False)
        save("baselines", m, summ, logs, s=s)
print(f"methods ok ({time.time()-T0:.0f}s)")

# 2. every attack against TrustFed and two robust rules, DP forced on
_ATK = ("trustfed", "multikrum", "flame", "fltrust", "foolsgold")
_sel = _ATK if STAGE in ("all", "attacks") else tuple(m for m in _ATK if STAGE == f"attacks:{m}")
for m in _sel:
    for atk, fr, st in (("label_flip", 0.34, 1), ("gaussian", 0.34, 1), ("sign_flip", 0.34, 1),
                        ("model_replacement", 0.34, 1), ("label_flip", 0.34, 2)):
        summ, _, logs = run_fl(fed, cfg, m, 42, attack=atk, byz_frac=fr,
                               attack_start=st, force_dp=True, verbose=False)
        save("byzantine", m, summ, logs, atk, fr, st, 42)
        if summ["byz_detect_rate_mean"] is None and m != "median":
            fails.append(f"{m}/{atk}: no detection metric")
print(f"attacks ok ({time.time()-T0:.0f}s)")

# 3. label-flip actually targets benign
from trustfed.attacks import label_flip
yy = label_flip(fed["clients"][0]["y_train"], fed["benign_idx"], 1.0, np.random.default_rng(0))
if not (yy == fed["benign_idx"]).all():
    fails.append("label flip did not map every attack label to benign")

# 4. raw preprocessing on fake raw files with the real column names / label strings
if run_stage("preprocess"):
    import tempfile
    tmp = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_raw_fixtures.py"),
                        os.path.join(tmp, "raw")], capture_output=True, text=True,
                       cwd=os.path.join(ROOT, "tests"))
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from trustfed.datasets import preprocess\n"
            "exp = {'CICIDS2017': 13, 'CICIoT2023': 34, 'Edge-IIoTset': 15}\n"
            "for ds, k in exp.items():\n"
            "    m = preprocess(ds, raw_dir=%r, cap=2000)\n"
            "    assert len(m['class_counts_final']) == k, (ds, len(m['class_counts_final']))\n"
            "    from trustfed.data import prepare_federation\n"
            "    f = prepare_federation(ds, 4, 0.5, 0, root_per_class=5)\n"
            "    assert f['train_test_shared_feature_vectors'] == 0, (ds, 'leak')\n"
            "print('PREPROCESS OK')\n") % (ROOT, os.path.join(tmp, "raw"))
    r2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                        env=dict(os.environ, TRUSTFED_HOME=os.path.join(tmp, "home")))
    print(r2.stdout[-400:], r2.stderr[-400:])
    if "PREPROCESS OK" not in r2.stdout:
        fails.append("raw preprocessing failed: " + r.stderr[-300:] + r2.stderr[-300:])

# 5. tables
if run_stage("tables"):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "make_tables.py"), "--results", R],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode:
        fails.append("make_tables failed")
    v = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "verify_audit.py"), os.path.join(R, "main")],
                       capture_output=True, text=True)
    print(v.stdout[-400:], v.stderr[-400:])
    if "INVALID" in v.stdout or "VALID" not in v.stdout:
        fails.append("verify_audit failed")
    nested = [d for d in os.listdir(os.path.join(R, "main")) if os.path.isdir(os.path.join(R, "main", d))]
    if nested:
        fails.append(f"unexpected sub-folders in results/main: {nested}")
print(f"\nStage '{STAGE}' {time.time()-T0:.0f}s")
print("SMOKE TEST", "PASSED" if not fails else "FAILED:\n  " + "\n  ".join(fails))
sys.exit(1 if fails else 0)

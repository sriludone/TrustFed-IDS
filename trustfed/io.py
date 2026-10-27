"""
io.py -- ONE result file per run, in a flat folder per experiment group:

    results/<group>/<dataset>__<method>__<attack>-f<frac>-s<start>[-eps<e>]__a<alpha>__seed<seed>.json
    results/main/...json  (+ one <same name>.audit.npz holding the run's SHAP vectors)

Merging results from several machines = copying all files into the same folders
(names never collide). A run is "done" when its .json exists; files are written
atomically, so an interrupted run never leaves a half-written result.
"""
import os, json, glob, tempfile


def run_name(dataset, method, attack="none", frac=0.0, start=1, alpha=0.5, seed=42, eps=None):
    e = f"-eps{eps}" if eps is not None else ""
    return f"{dataset}__{method}__{attack}-f{frac}-s{start}{e}__a{alpha}__seed{seed}"


def run_path(results, group, name):
    return os.path.join(results, group, name + ".json")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=float)
    os.replace(tmp, path)


def load_runs(results):
    runs = []
    for f in sorted(glob.glob(os.path.join(results, "*", "*.json"))):
        if os.sep + "tables" + os.sep in f:
            continue
        try:
            r = json.load(open(f)); r["_file"] = f; runs.append(r)
        except Exception as e:
            print(f"  [skip unreadable] {f}: {e}")
    return runs

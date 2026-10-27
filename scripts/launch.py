"""Start the full campaign on THIS machine: one process per GPU, each running its shard
through every experiment group in order. Same command on Linux and Windows.

  python scripts/launch.py --first-k 0 --num-gpus 4 --total 20
  python scripts/launch.py --first-k 0 --num-gpus 4 --total 20 --dry-run   # show the plan only

--first-k : shard number of this machine's first GPU (GPUs on machines numbered before it)
--total   : total GPUs used across ALL machines (identical everywhere)
Logs: logs/shard_<K>.log. Finished runs are skipped, so re-running resumes."""
import os, sys, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# seeds: 5 for main and baselines, 3 for the sweeps (defaults in run_experiments.py)
GROUPS = [["--group", "main"],
          ["--group", "baselines"],
          ["--group", "privacy"],
          ["--group", "ablation"],
          ["--group", "alpha"],
          ["--group", "byzantine", "--byz-datasets", "Edge-IIoTset", "CICIDS2017"]]

ap = argparse.ArgumentParser()
ap.add_argument("--first-k", type=int, required=True)
ap.add_argument("--num-gpus", type=int, required=True)
ap.add_argument("--total", type=int, required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--results", default="results")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--quick", action="store_true", help="5 rounds per run; pipeline test only")
a = ap.parse_args()

os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
threads = str(max(1, (os.cpu_count() or 1) // a.num_gpus))
procs = []
for g in range(a.num_gpus):
    k = a.first_k + g
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONWARNINGS="ignore", OMP_NUM_THREADS=threads)
    cmds = [[sys.executable, os.path.join(ROOT, "run_experiments.py"), *grp, "--device", a.device,
             "--results", a.results, "--shard", f"{k}/{a.total}"] + (["--dry-run"] if a.dry_run else []) + (["--quick"] if a.quick else [])
            for grp in GROUPS]
    if a.dry_run:
        print(f"--- GPU {g} -> shard {k}/{a.total}")
        for c in cmds:
            out = subprocess.run(c, cwd=ROOT, env=env, capture_output=True, text=True).stdout.splitlines()
            print(f"  {c[3]:10s}: {out[0] if out else '?'}")
        continue
    log = open(os.path.join(ROOT, "logs", f"shard_{k}.log"), "a")
    script = " && ".join(subprocess.list2cmdline(c) if os.name == "nt" else " ".join(f"'{x}'" for x in c) for c in cmds)
    procs.append(subprocess.Popen(script, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    print(f"started GPU {g} -> shard {k}/{a.total}  (log: logs/shard_{k}.log)")
for p in procs:
    p.wait()
if procs:
    print("ALL SHARDS ON THIS MACHINE FINISHED")

"""raw -> data/processed (exact class definitions, identifier drop, one-hot, dedup, per-class cap).

Expected raw layout (any sub-folder depth):
  data/raw/Edge-IIoTset/**/DNN-EdgeIIoT-dataset.csv
  data/raw/CICIoT2023/**/*.csv            (the CSV release, 'label' column, 34 classes)
  data/raw/CICIDS2017/**/*.csv            (MachineLearningCVE or TrafficLabelling CSVs)

  python scripts/preprocess.py --all
  python scripts/preprocess.py --dataset CICIoT2023 --cap 25000
  python scripts/preprocess.py --dataset CICIoT2023 --drop-extra IAT   # leakage check (see README)
Then open data/processed/<ds>_meta.json and check the dropped columns and dedup counts."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trustfed.datasets import preprocess

ap = argparse.ArgumentParser()
ap.add_argument("--dataset"); ap.add_argument("--all", action="store_true")
ap.add_argument("--raw-dir"); ap.add_argument("--cap", type=int)
ap.add_argument("--keep", nargs="*", default=[]); ap.add_argument("--drop-extra", nargs="*", default=[])
ap.add_argument("--dedup", choices=["keep", "drop"], default="keep",
                help="keep: duplicates kept, group-aware split prevents leakage (default); drop: remove them")
a = ap.parse_args()
for ds in (["Edge-IIoTset", "CICIoT2023", "CICIDS2017"] if a.all else [a.dataset]):
    preprocess(ds, raw_dir=a.raw_dir, cap=a.cap, keep=tuple(a.keep), drop_extra=tuple(a.drop_extra), dedup=a.dedup)

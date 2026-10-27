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

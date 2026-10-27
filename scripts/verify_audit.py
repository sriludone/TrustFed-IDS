"""Re-verify the explanation audit chains of finished main runs from their two files.
   python scripts/verify_audit.py results/main"""
import os, sys, json, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trustfed.audit import verify_chain
folder = sys.argv[1] if len(sys.argv) > 1 else "results/main"
for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
    r = json.load(open(f))
    if "audit_chain" not in r:
        continue
    vec = dict(np.load(f[:-5] + ".audit.npz"))
    t0 = time.time()
    res = {c: verify_chain(e, vec, r["audit_chain"]["anchors"][c]) for c, e in r["audit_chain"]["entries"].items()}
    ok = all(v["valid"] for v in res.values())
    print(f"{os.path.basename(f)}: {'VALID' if ok else 'INVALID'} ({len(res)} chains, "
          f"{sum(v.get('entries', 0) for v in res.values())} entries, {time.time()-t0:.2f}s)")

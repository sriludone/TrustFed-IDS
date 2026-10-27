"""
audit.py -- per-client explanation hash chains with server-side anchoring.

Trust model (state this in the paper):
  * Each CLIENT keeps its own append-only chain. Entry t commits to
    SHA-256(SHAP vector) and to entry t-1. The client keeps the SHAP vector so the
    digest can be recomputed at audit time.
  * Each round the SERVER records the client's new chain head (anchoring).
  * An auditor detects (a) any edit to a stored SHAP vector or chain entry, by
    recomputation, and (b) a client rewriting its whole chain after submission, by
    comparing heads with the server's anchors. Collusion of client AND server, or a
    client lying at computation time, is NOT detected.

Storage: chain entries and anchors go into the run's single JSON file; all SHAP
vectors of a run go into ONE compressed file <run>.audit.npz (key c<client>_r<round>).
scripts/verify_audit.py re-verifies a finished run from those two files.
"""
import copy, json, hashlib, time
import numpy as np

GENESIS = "0" * 64


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def _entry_hash(e):
    body = {k: e[k] for k in ("cid", "round", "shap_digest", "shap_status", "dp", "prev")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def vkey(cid, rnd):
    return f"c{cid}_r{rnd}"


class AuditLedger:
    """All clients' chains for one run, kept in memory and saved once at the end."""

    def __init__(self):
        self.entries, self.vectors, self.heads, self.anchors = {}, {}, {}, {}

    def append(self, cid, rnd, shap_vec, status, dp):
        t0 = time.time()
        prev = self.heads.get(cid, GENESIS)
        v = np.ascontiguousarray(shap_vec, dtype=np.float32)
        e = {"cid": int(cid), "round": int(rnd), "shap_digest": digest(v),
             "shap_status": status, "dp": bool(dp), "prev": prev}
        e["hash"] = _entry_hash(e)
        self.entries.setdefault(cid, []).append(e)
        self.vectors[vkey(cid, rnd)] = v
        self.heads[cid] = e["hash"]
        self.anchors.setdefault(cid, {})[str(rnd)] = e["hash"]       # server records the head
        return e["hash"], time.time() - t0

    def save_vectors(self, path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **self.vectors)

    def to_json(self):
        return {"entries": {str(k): v for k, v in self.entries.items()},
                "anchors": {str(k): v for k, v in self.anchors.items()}}


def verify_chain(entries, vectors, anchors=None):
    prev = GENESIS
    for e in entries:
        if e["prev"] != prev:
            return {"valid": False, "round": e["round"], "reason": "broken link"}
        if _entry_hash(e) != e["hash"]:
            return {"valid": False, "round": e["round"], "reason": "entry altered"}
        if digest(vectors[vkey(e["cid"], e["round"])]) != e["shap_digest"]:
            return {"valid": False, "round": e["round"], "reason": "explanation altered"}
        if anchors is not None and anchors.get(str(e["round"])) != e["hash"]:
            return {"valid": False, "round": e["round"], "reason": "head != server anchor"}
        prev = e["hash"]
    return {"valid": True, "entries": len(entries), "head": prev}


def tamper_suite(entries, vectors, anchors):
    """Applies each tampering scenario to a COPY of one client's chain and reports
    whether verification detects it."""
    res = []
    mid = entries[len(entries) // 2]
    k = vkey(mid["cid"], mid["round"])

    def one_cell(v):
        v = v.copy(); v[0, 0] += 1e-3; return v
    scenarios = {"single feature +0.001": one_cell,
                 "all features x1.001": lambda v: v * 1.001,
                 "feature order shuffled": lambda v: v[:, ::-1].copy(),
                 "no modification (control)": lambda v: v}
    for name, fn in scenarios.items():
        vec = dict(vectors); vec[k] = fn(vectors[k]).astype(np.float32)
        t0 = time.time(); r = verify_chain(entries, vec, anchors)
        res.append({"scenario": name, "detected": not r["valid"], "verify_sec": round(time.time() - t0, 5)})
    ent, vec, prev = copy.deepcopy(entries), dict(vectors), GENESIS
    for e in ent:
        kk = vkey(e["cid"], e["round"]); vec[kk] = (vectors[kk] * 0.5).astype(np.float32)
        e["shap_digest"], e["prev"] = digest(vec[kk]), prev
        e["hash"] = _entry_hash(e); prev = e["hash"]
    res.append({"scenario": "client rewrites whole chain (no anchors)",
                "detected": not verify_chain(ent, vec, None)["valid"]})
    res.append({"scenario": "client rewrites whole chain (with server anchors)",
                "detected": not verify_chain(ent, vec, anchors)["valid"]})
    return res

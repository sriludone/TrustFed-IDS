"""
metrics.py -- evaluation metrics.

Two false-positive measures are reported because they answer different questions:
  fpr_macro   : mean one-vs-rest FPR over classes (what the old code reported).
  benign_far  : fraction of BENIGN flows predicted as ANY attack. This is the
                number the paper's "flows wrongly acted on per day" argument needs.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)


def metrics_from_confusion(cm, benign_idx=None):
    cm = np.asarray(cm, dtype=float)
    tot = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(1)
    pred = cm.sum(0)
    present = support > 0
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    fp = pred - tp
    tn = tot - support - pred + tp
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(tp), where=(fp + tn) > 0)
    out = {
        "accuracy": float(tp.sum() / tot) if tot else 0.0,
        "balanced_accuracy": float(rec[present].mean()) if present.any() else 0.0,
        "f1_macro": float(f1[present].mean()) if present.any() else 0.0,
        "precision_macro": float(prec[present].mean()) if present.any() else 0.0,
        "fpr_macro": float(fpr[present].mean()) if present.any() else 0.0,
    }
    if benign_idx is not None and support[benign_idx] > 0:
        out["benign_far"] = float(1 - cm[benign_idx, benign_idx] / support[benign_idx])
        att = [i for i in range(len(cm)) if i != benign_idx]
        att_tot = cm[att].sum()
        out["attack_detection_rate"] = float(1 - cm[att, benign_idx].sum() / att_tot) if att_tot else 0.0
    return out


def evaluate(proba, y, n_classes, benign_idx=None, with_report=False):
    pred = proba.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    m = metrics_from_confusion(cm, benign_idx)
    try:
        present = np.unique(y)
        if len(present) == 2 and n_classes == 2:
            m["auc_roc"] = float(roc_auc_score(y, proba[:, 1]))
        elif len(present) > 2:
            P = proba[:, present]
            P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
            m["auc_roc"] = float(roc_auc_score(y, P, multi_class="ovr", labels=present,
                                               average="macro"))
    except Exception:
        m["auc_roc"] = float("nan")
    if with_report:
        support = cm.sum(1)
        tp = np.diag(cm)
        rec = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
        pr = np.divide(tp, cm.sum(0), out=np.zeros(n_classes), where=cm.sum(0) > 0)
        f1 = np.divide(2 * pr * rec, pr + rec, out=np.zeros(n_classes), where=(pr + rec) > 0)
        m["per_class"] = {"precision": pr.tolist(), "recall": rec.tolist(),
                          "f1": f1.tolist(), "support": support.astype(int).tolist()}
        m["confusion_matrix"] = cm.astype(int).tolist()
    return m, cm

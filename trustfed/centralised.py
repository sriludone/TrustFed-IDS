"""
centralised.py -- non-federated reference models (Centralised ResidualMLP, LSTM-IDS).
Same data as the FL runs: train = union of client train splits, validation =
union of client val splits, identical test set. Epoch selected on VALIDATION.
"""
import os, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from .models import get_model, get_weights, set_weights, count_parameters, predict_proba
from .client import sqrt_inv_freq_weights
from .metrics import evaluate


def run_centralised(fed, cfg, seed, arch="residual_mlp", epochs=None, verbose=True):
    """Returns (summary, best_weights, compact_epoch_log); writes nothing."""
    t_run = time.time(); torch.manual_seed(seed)
    X = np.concatenate([c["X_train"] for c in fed["clients"]]); y = np.concatenate([c["y_train"] for c in fed["clients"]])
    Xv = np.concatenate([c["X_val"] for c in fed["clients"]]); yv = np.concatenate([c["y_val"] for c in fed["clients"]])
    nf, nc, b = fed["n_features"], fed["n_classes"], fed["benign_idx"]
    dev = cfg.get("device", "cpu")
    model = get_model(nf, nc, arch, cfg["hidden"], cfg["n_blocks"]).to(dev)
    cw = torch.tensor(sqrt_inv_freq_weights(y, nc), device=dev)
    sw = np.zeros(len(y), np.float32)
    for c, k in zip(*np.unique(y, return_counts=True)):
        sw[y == c] = 1 / np.sqrt(k)
    dl = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=cfg["batch_size"],
                    sampler=WeightedRandomSampler(torch.from_numpy(sw), len(y), True,
                                                  generator=torch.Generator().manual_seed(seed)))
    epochs = epochs or cfg.get("centralised_epochs", 30)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_w, logs, best_ep = -1, None, [], 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            (F.cross_entropy(model(xb), yb, reduction="none") * cw[yb]).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        mv, _ = evaluate(predict_proba(model, Xv, device=dev), yv, nc, b)
        logs.append({"round": ep, "val_balacc": round(mv["balanced_accuracy"], 4), "val_f1": round(mv["f1_macro"], 4)})
        if verbose:
            print(f"  [central-{arch}|{fed['dataset']}|s{seed}] ep{ep} valBA={mv['balanced_accuracy']:.4f}", flush=True)
        if mv["balanced_accuracy"] > best:
            best, best_w, best_ep = mv["balanced_accuracy"], get_weights(model), ep
    set_weights(model, best_w)
    final, _ = evaluate(predict_proba(model, fed["X_test"], device=dev), fed["y_test"], nc, b, with_report=True)
    name = "centralised" if arch == "residual_mlp" else "lstm_ids"
    summary = {"dataset": fed["dataset"], "method": name, "seed": seed, "attack": "none", "byz_frac": 0.0,
               "use_dp": False, "alpha": fed["alpha"], "best_val_round": best_ep,
               "best_val_balanced_accuracy": best, "test": final, "n_params": count_parameters(model),
               "runtime_sec": round(time.time() - t_run, 1), "class_names": fed["class_names"],
               "dropped_classes": fed.get("dropped_classes", {}),
               "train_test_shared_feature_vectors": fed.get("train_test_shared_feature_vectors")}
    return summary, best_w, logs

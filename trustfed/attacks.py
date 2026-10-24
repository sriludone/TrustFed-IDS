"""
attacks.py -- poisoning attacks. All use local RNGs (no global reseeding).

Data-level   : label_flip -> attack labels flipped to the BENIGN index (the old
               code flipped to class 0, which is benign only for CICIDS2017).
Update-level : gaussian    -> delta + N(0, (scale*std(delta))^2)
               sign_flip   -> -scale * delta
               model_replacement -> label-flip training + update boosted by the
                              number of participants (Bagdasaryan et al. style)
Adaptive     : any attack starting at round `start_round` (honest before), to
               test the EMA trust term against "honest-then-poison" clients.
"""
import numpy as np

ATTACKS = ("none", "label_flip", "gaussian", "sign_flip", "model_replacement")


def label_flip(y, benign_idx, flip_rate, rng):
    y = y.copy()
    att = np.where(y != benign_idx)[0]
    k = int(len(att) * flip_rate)
    if k:
        y[rng.choice(att, k, replace=False)] = benign_idx
    return y


def needs_flipped_data(attack):
    return attack in ("label_flip", "model_replacement")


def poison_update(delta, attack, rng, scale=5.0, n_participants=15):
    if attack == "gaussian":
        return delta + rng.normal(0, scale * (delta.std() + 1e-12), size=delta.shape).astype(delta.dtype)
    if attack == "sign_flip":
        return (-scale * delta).astype(delta.dtype)
    if attack == "model_replacement":
        return (n_participants * delta).astype(delta.dtype)
    return delta


def is_active(attack, rnd, start_round=1):
    return attack != "none" and rnd >= start_round

"""
explain.py -- SHAP explanations for the audit chain.

The old code ran KernelExplainer with nsamples=20 for 76 features (far too few
for a meaningful estimate) and, when SHAP failed, silently hashed a timestamp.
Here: GradientExplainer by default (fast enough for the Pi, seeded), Kernel
SHAP optional with an adequate sample budget, and an explicit status field.
The returned array is the SHAP value of the PREDICTED class for each reference
row, shape (k, d). It is stored by the client so a hash can be re-verified.
"""
import time
import numpy as np
import torch


def expected_gradients(model, X_ref, X_bg, n_samples=64, seed=0):
    """Expected Gradients (Erion et al., 2021) -- the SHAP estimator implemented by
    shap.GradientExplainer -- for the PREDICTED class, computed in ONE batched backward
    pass. shap's implementation loops sample-by-sample over every output class
    (~10k backward passes per round here), which dominated round time.
    attr_j(x) = E_{b~bg, a~U(0,1)} [ (x_j - b_j) * d f_c(b + a(x-b)) / d z_j ]."""
    g = torch.Generator().manual_seed(seed)
    model.eval()
    X = torch.as_tensor(X_ref, dtype=torch.float32); B = torch.as_tensor(X_bg, dtype=torch.float32)
    k, d = X.shape
    with torch.no_grad():
        pred = model(X).argmax(1)
    idx = torch.randint(len(B), (k, n_samples), generator=g)
    a = torch.rand(k, n_samples, 1, generator=g)
    b = B[idx]                                   # (k, n, d)
    diff = X[:, None, :] - b
    z = (b + a * diff).reshape(-1, d).requires_grad_(True)
    out = model(z)
    sel = out.gather(1, pred.repeat_interleave(n_samples)[:, None]).sum()
    (grad,) = torch.autograd.grad(sel, z)
    return (diff * grad.reshape(k, n_samples, d)).mean(1).detach().numpy().astype(np.float32)


def shap_for_predicted(model, X_ref, X_bg, method="expected_gradients", seed=0, kernel_nsamples=512):
    t0 = time.time()
    try:
        if method == "expected_gradients":
            return expected_gradients(model, X_ref, X_bg, seed=seed), "ok", time.time() - t0
        import shap
        np.random.seed(seed); torch.manual_seed(seed)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X_ref)).argmax(1).numpy()
        if method == "gradient":
            ex = shap.GradientExplainer(model, torch.from_numpy(X_bg))
            sv = ex.shap_values(torch.from_numpy(X_ref), nsamples=64, rseed=seed)
        else:
            def f(x):
                with torch.no_grad():
                    return torch.softmax(model(torch.from_numpy(x.astype(np.float32))), -1).numpy()
            ex = shap.KernelExplainer(f, X_bg)
            sv = ex.shap_values(X_ref, nsamples=kernel_nsamples, silent=True)
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[0] == len(X_ref):        # (k, d, C)
            out = sv[np.arange(len(X_ref)), :, pred]
        else:                                                  # (C, k, d)
            out = sv[pred, np.arange(len(X_ref)), :]
        return out.astype(np.float32), "ok", time.time() - t0
    except Exception as e:  # recorded, never hidden
        return np.zeros((len(X_ref), X_ref.shape[1]), np.float32), f"error:{type(e).__name__}", time.time() - t0

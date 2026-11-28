from __future__ import annotations

import numpy as np
import pandas as pd


def feature_noise(
    X: pd.DataFrame,
    eps: float = 0.5,
    seed: int = 0,
    scale_ref: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Bounded Gaussian feature noise.

    Noise SD is ``eps * column_SD(scale_ref)``. When ``scale_ref`` is the
    training matrix (recommended), this matches the train-only statistics protocol.
    If ``scale_ref`` is omitted, falls back to ``X`` (legacy behavior).
    """
    rng = np.random.default_rng(seed)
    out = X.copy().astype(float)
    ref = scale_ref if scale_ref is not None else out
    sd = ref.astype(float).std(ddof=0).replace(0, 1.0)
    # align columns
    sd = sd.reindex(out.columns).fillna(1.0)
    noise = rng.normal(0, 1.0, size=out.shape) * (eps * sd.to_numpy())
    out = out + noise
    # Keep aggregate min/max consistent after independent noise (avoids min > max rows).
    for col in list(out.columns):
        if not col.endswith("_min"):
            continue
        max_col = col[: -len("_min")] + "_max"
        if max_col not in out.columns:
            continue
        lo = out[col].to_numpy()
        hi = out[max_col].to_numpy()
        new_lo = np.minimum(lo, hi)
        new_hi = np.maximum(lo, hi)
        out[col] = new_lo
        out[max_col] = new_hi
    return out


def missingness(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cols:
        if c in out.columns:
            out[c] = np.nan
    return out


def label_poison(y: np.ndarray, rate: float, seed: int = 0) -> np.ndarray:
    """Uniform random label flips (not targeted/backdoor poisoning)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int).copy()
    k = int(round(rate * len(y)))
    if k <= 0:
        return y
    idx = rng.choice(len(y), size=k, replace=False)
    y[idx] = 1 - y[idx]
    return y


def label_poison_targeted(
    y: np.ndarray,
    p_train: np.ndarray,
    rate: float,
    seed: int = 0,
) -> np.ndarray:
    """Flip labels of the most confidently correct training rows (targeted baseline)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int).copy()
    p = np.asarray(p_train).astype(float)
    k = int(round(rate * len(y)))
    if k <= 0:
        return y
    conf_correct = np.where(y == 1, p, 1.0 - p)
    # break ties with RNG for determinism
    order = np.argsort(-conf_correct + rng.uniform(0, 1e-9, size=len(y)))
    idx = order[:k]
    y[idx] = 1 - y[idx]
    return y


def fgsm_feature_shift(
    X: pd.DataFrame,
    coef: np.ndarray,
    scale_ref: pd.DataFrame,
    eps: float = 0.5,
    direction: str = "increase_risk",
) -> pd.DataFrame:
    """Closed-form FGSM-style shift for a linear model on standardized features.

    Applies ``+/- eps * train_SD * sign(coef)`` in original feature space
    (equivalent to an L_inf step of size ``eps`` on z-scored inputs).
    """
    out = X.copy().astype(float)
    coef = np.asarray(coef, dtype=float).ravel()
    if coef.shape[0] != out.shape[1]:
        raise ValueError("coef length must match number of features")
    sd = scale_ref.astype(float).std(ddof=0).replace(0, 1.0)
    sd = sd.reindex(out.columns).fillna(1.0).to_numpy()
    step = eps * sd * np.sign(coef)
    if direction == "decrease_risk":
        step = -step
    elif direction != "increase_risk":
        raise ValueError(direction)
    return out + step

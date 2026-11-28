from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def summarize(y_true, y_prob) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out = {
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "mean_pred": float(np.mean(y_prob)) if len(y_prob) else float("nan"),
        "calib_in_the_large": float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "brier": float("nan"),
        "calib_slope": float("nan"),
        "calib_intercept": float("nan"),
    }
    if len(y_true) < 10 or y_true.min() == y_true.max():
        if len(y_true):
            out["calib_in_the_large"] = float(np.mean(y_prob) - np.mean(y_true))
        return out
    out["auroc"] = float(roc_auc_score(y_true, y_prob))
    out["auprc"] = float(average_precision_score(y_true, y_prob))
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["calib_in_the_large"] = float(np.mean(y_prob) - np.mean(y_true))
    cal = calibration_slope_intercept(y_true, y_prob)
    out.update(cal)
    return out


def calibration_slope_intercept(y_true, y_prob) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)
    if len(y_true) < 30 or y_true.min() == y_true.max():
        return {"calib_slope": float("nan"), "calib_intercept": float("nan")}
    logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(logit, y_true)
    return {
        "calib_slope": float(lr.coef_[0, 0]),
        "calib_intercept": float(lr.intercept_[0]),
    }


def _metric_only(y_true, y_prob, metric: str) -> float:
    """Compute one metric without refitting calibration (fast bootstrap path)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) < 2 or y_true.min() == y_true.max():
        return float("nan")
    if metric == "auroc":
        return float(roc_auc_score(y_true, y_prob))
    if metric == "auprc":
        return float(average_precision_score(y_true, y_prob))
    if metric == "brier":
        return float(brier_score_loss(y_true, y_prob))
    return float(summarize(y_true, y_prob)[metric])


def bootstrap_metric_ci(
    y_true, y_prob, metric: str = "auroc", n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.min() == yt.max():
            continue
        m = _metric_only(yt, yp, metric)
        if np.isfinite(m):
            vals.append(m)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_bootstrap_delta_ci(
    y_true,
    y_prob_a,
    y_prob_b,
    metric: str = "auroc",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap CI for metric(b) - metric(a) on the same resampled indices."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    ya = np.asarray(y_prob_a).astype(float)
    yb = np.asarray(y_prob_b).astype(float)
    n = len(y_true)
    point_a = _metric_only(y_true, ya, metric)
    point_b = _metric_only(y_true, yb, metric)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        ma = _metric_only(yt, ya[idx], metric)
        mb = _metric_only(yt, yb[idx], metric)
        if np.isfinite(ma) and np.isfinite(mb):
            deltas.append(mb - ma)
    if not deltas:
        return {
            "metric": metric,
            "delta": float(point_b - point_a) if np.isfinite(point_a) and np.isfinite(point_b) else float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_boot_used": 0.0,
        }
    arr = np.asarray(deltas)
    return {
        "metric": metric,
        "delta": float(point_b - point_a),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n_boot_used": float(len(arr)),
    }


def risk_coverage_curve(
    y_true,
    y_prob,
    rank_scores,
    coverages=(1.0, 0.9, 0.8, 0.7, 0.6),
    keep: str = "lowest",
) -> list[dict]:
    """Selective prediction by ranking score.

    ``keep="lowest"`` retains the lowest-score fraction (anomaly gate).
    ``keep="highest"`` retains the highest-score fraction (model confidence).
    Reports Brier, event rate kept, and fraction of all events abstained.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    scores = np.asarray(rank_scores).astype(float)
    order = np.argsort(scores)
    if keep == "highest":
        order = order[::-1]
    elif keep != "lowest":
        raise ValueError(keep)
    rows = []
    n = len(y_true)
    events_total = float(y_true.sum())
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        idx = order[:k]
        m = summarize(y_true[idx], y_prob[idx])
        events_kept = float(y_true[idx].sum())
        frac_events_abstained = (
            float((events_total - events_kept) / events_total) if events_total > 0 else float("nan")
        )
        rows.append(
            {
                "coverage": float(cov),
                "n_kept": int(k),
                "event_rate_kept": m["event_rate"],
                "frac_events_abstained": frac_events_abstained,
                "brier": m["brier"],
                "auroc": m["auroc"],
            }
        )
    return rows

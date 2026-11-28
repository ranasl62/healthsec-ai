#!/usr/bin/env python3
"""Multi-split / multi-noise and dose-response sensitivity (AUROC point estimates)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from healthsec_ai.adapters import get_adapter
from healthsec_ai.attacks.engine import feature_noise, missingness
from healthsec_ai.runner import _importance_topk, _model

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = ROOT / "results"


def _auroc(y, p) -> float:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def multi_split_noise(
    dataset: str,
    n_splits: int = 20,
    base_seed: int = 20261126,
    models: list[str] | None = None,
    eps: float = 0.5,
) -> pd.DataFrame:
    models = models or ["logreg", "logreg_unweighted", "rf", "xgboost"]
    bundle = get_adapter(dataset).load(DATA)
    X, y = bundle.X, bundle.y
    rows = []
    for i in range(n_splits):
        seed = base_seed + i
        noise_seed = base_seed + 1000 + i
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        yte_np = yte.to_numpy()
        for mname in models:
            clf = _model(mname, seed)
            clf.fit(Xtr, ytr)
            p_clean = clf.predict_proba(Xte)[:, 1]
            Xn = feature_noise(Xte, eps=eps, seed=noise_seed, scale_ref=Xtr)
            p_noise = clf.predict_proba(Xn)[:, 1]
            top = _importance_topk(clf, list(X.columns), k=3)
            p_miss = clf.predict_proba(missingness(Xte, top))[:, 1]
            ac, an, am = _auroc(yte_np, p_clean), _auroc(yte_np, p_noise), _auroc(yte_np, p_miss)
            rows.append(
                {
                    "dataset": dataset,
                    "split_seed": seed,
                    "noise_seed": noise_seed,
                    "model": mname,
                    "auroc_clean": ac,
                    "auroc_noise": an,
                    "auroc_miss": am,
                    "delta_noise": an - ac,
                    "delta_miss": am - ac,
                    "brier_clean": float(np.mean((p_clean - yte_np) ** 2)),
                    "brier_noise": float(np.mean((p_noise - yte_np) ** 2)),
                    "brier_miss": float(np.mean((p_miss - yte_np) ** 2)),
                }
            )
    return pd.DataFrame(rows)


def dose_response(
    dataset: str,
    seed: int = 20261126,
    models: list[str] | None = None,
    eps_grid: list[float] | None = None,
) -> pd.DataFrame:
    models = models or ["logreg", "logreg_unweighted", "rf", "xgboost"]
    eps_grid = eps_grid or [0.1, 0.25, 0.5, 0.75, 1.0]
    bundle = get_adapter(dataset).load(DATA)
    X, y = bundle.X, bundle.y
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    yte_np = yte.to_numpy()
    rows = []
    for mname in models:
        clf = _model(mname, seed)
        clf.fit(Xtr, ytr)
        p_clean = clf.predict_proba(Xte)[:, 1]
        ac = _auroc(yte_np, p_clean)
        for eps in eps_grid:
            Xn = feature_noise(Xte, eps=eps, seed=seed, scale_ref=Xtr)
            an = _auroc(yte_np, clf.predict_proba(Xn)[:, 1])
            rows.append(
                {
                    "dataset": dataset,
                    "model": mname,
                    "eps": eps,
                    "auroc_clean": ac,
                    "auroc_noise": an,
                    "delta_noise": an - ac,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = ["synthetic_cv", "framingham", "cinc2012"]
    multi_frames = []
    dose_frames = []
    for ds in datasets:
        print("multi-split", ds)
        m = multi_split_noise(ds, n_splits=20)
        m.to_csv(OUT / f"sensitivity_multisplit_{ds}.csv", index=False)
        multi_frames.append(m)
        print("dose-response", ds)
        d = dose_response(ds)
        d.to_csv(OUT / f"sensitivity_dose_{ds}.csv", index=False)
        dose_frames.append(d)
    summary = (
        pd.concat(multi_frames)
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            n=("delta_noise", "count"),
            delta_noise_mean=("delta_noise", "mean"),
            delta_noise_sd=("delta_noise", "std"),
            delta_miss_mean=("delta_miss", "mean"),
            delta_miss_sd=("delta_miss", "std"),
            brier_miss_minus_clean=("brier_miss", "mean"),
            brier_clean_mean=("brier_clean", "mean"),
        )
    )
    # brier under missingness minus clean mean across splits
    raw = pd.concat(multi_frames)
    b = (
        raw.assign(brier_delta_miss=lambda t: t["brier_miss"] - t["brier_clean"])
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            brier_delta_miss_mean=("brier_delta_miss", "mean"),
            brier_delta_miss_sd=("brier_delta_miss", "std"),
        )
    )
    summary = summary.merge(b, on=["dataset", "model"])
    summary.to_csv(OUT / "sensitivity_multisplit_summary.csv", index=False)
    pd.concat(dose_frames).to_csv(OUT / "sensitivity_dose_all.csv", index=False)
    meta = {
        "n_splits": 20,
        "eps_grid": [0.1, 0.25, 0.5, 0.75, 1.0],
        "models": ["logreg", "logreg_unweighted", "rf", "xgboost"],
        "note": "Point-estimate AUROC/Brier across splits; not bootstrap CIs",
    }
    (OUT / "sensitivity.meta.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

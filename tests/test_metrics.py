"""Metric invariants and paired-bootstrap smoke tests."""
from __future__ import annotations

import numpy as np

from healthsec_ai.evaluation.metrics import (
    paired_bootstrap_delta_ci,
    risk_coverage_curve,
    summarize,
)


def test_summarize_perfect_ranking():
    y = np.array([0] * 20 + [1] * 20)
    p = np.concatenate([np.linspace(0.05, 0.45, 20), np.linspace(0.55, 0.95, 20)])
    m = summarize(y, p)
    assert m["auroc"] == 1.0
    assert m["brier"] < 0.25


def test_paired_delta_zero_when_identical():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(0.1, 0.9, 200)
    d = paired_bootstrap_delta_ci(y, p, p, n_boot=200, seed=1)
    assert abs(d["delta"]) < 1e-12
    assert d["ci_lo"] <= 0 <= d["ci_hi"]


def test_risk_coverage_monotone_n_kept():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 100)
    p = rng.uniform(0, 1, 100)
    scores = rng.normal(size=100)
    rows = risk_coverage_curve(y, p, scores)
    ns = [r["n_kept"] for r in rows]
    assert ns == sorted(ns, reverse=True)
    assert "frac_events_abstained" in rows[0]


def test_feature_noise_uses_train_scale():
    import pandas as pd
    from healthsec_ai.attacks.engine import feature_noise

    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(0, 10, 200), "b": rng.normal(0, 1, 200)})
    test = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)})
    noisy = feature_noise(test, eps=0.5, seed=1, scale_ref=train)
    # With train SD≈10 on column a, noise magnitude should exceed test-only scaling
    legacy = feature_noise(test, eps=0.5, seed=1, scale_ref=None)
    assert abs((noisy["a"] - test["a"]).std()) > abs((legacy["a"] - test["a"]).std())

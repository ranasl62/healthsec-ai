"""v0.2.1 stress / adapter regressions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from healthsec_ai.attacks.engine import (
    feature_noise,
    fgsm_feature_shift,
    label_poison_targeted,
)
from healthsec_ai.adapters.cinc2012 import _parse_record


def test_min_max_consistency_after_noise():
    X = pd.DataFrame({"a_min": [1.0, 2.0], "a_max": [3.0, 4.0], "b": [0.0, 1.0]})
    out = feature_noise(X, eps=5.0, seed=0, scale_ref=X)
    assert (out["a_min"] <= out["a_max"]).all()


def test_fgsm_moves_in_coef_direction():
    X = pd.DataFrame({"x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0]})
    coef = np.array([1.0, -1.0])
    out = fgsm_feature_shift(X, coef, X, eps=0.5)
    assert out["x"].iloc[0] > 0
    assert out["y"].iloc[0] < 0


def test_targeted_flips_prefer_confident():
    y = np.array([0, 0, 1, 1])
    # confidently correct: row1 (y=0,p=0.1) and row2 (y=1,p=0.9)
    p = np.array([0.9, 0.1, 0.9, 0.1])
    out = label_poison_targeted(y, p, rate=0.5, seed=0)
    assert out[1] == 1 and out[2] == 0


def test_cinc_parse_recodes_minus_one(tmp_path):
    fp = tmp_path / "1.txt"
    fp.write_text(
        "Time,Parameter,Value\n"
        "00:00,RecordID,1\n"
        "00:00,Age,-1\n"
        "00:00,Gender,1\n"
        "00:00,Height,170\n"
        "00:00,Weight,70\n"
        "00:00,ICUType,3\n"
        "00:01,HR,80\n"
        "00:02,HR,90\n"
    )
    row = _parse_record(fp)
    assert np.isnan(row["Age"])
    assert row["HR_min"] <= row["HR_max"]

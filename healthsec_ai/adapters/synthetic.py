from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseAdapter, TaskBundle


class SyntheticCVAdapter(BaseAdapter):
    name = "synthetic_cv"

    def load(self, root: Path, n: int = 4000, seed: int = 20261126) -> TaskBundle:
        rng = np.random.default_rng(seed)
        z = rng.normal(0, 1, n)
        X = pd.DataFrame(
            {
                "age": np.clip(65 + 10 * z + rng.normal(0, 8, n), 18, 95),
                "sex_male": rng.binomial(1, 0.55, n),
                "sbp": np.clip(120 - 8 * z + rng.normal(0, 15, n), 70, 220),
                "hr": np.clip(80 + 6 * z + rng.normal(0, 12, n), 40, 180),
                "creatinine": np.clip(1.1 + 0.4 * z + rng.normal(0, 0.3, n), 0.3, 8),
                "lactate": np.clip(1.5 + 0.8 * z + rng.normal(0, 0.5, n), 0.4, 15),
                "spo2": np.clip(96 - 2 * z + rng.normal(0, 2, n), 70, 100),
                "wbc": np.clip(9 + 2 * z + rng.normal(0, 3, n), 1, 40),
            }
        )
        logit = -2.2 + 1.1 * z + 0.4 * (X["lactate"] - 1.5) - 0.02 * (X["sbp"] - 120)
        p = 1 / (1 + np.exp(-logit))
        y = pd.Series(rng.binomial(1, p), name="y")
        return TaskBundle(
            name=self.name,
            X=X,
            y=y,
            feature_names=list(X.columns),
            provenance={
                "source": "HealthSec-AI synthetic CV-ICU generator",
                "n": int(n),
                "events": int(y.sum()),
                "seed": seed,
                "note": "SYNTHETIC — reproducible fixtures only",
            },
        )

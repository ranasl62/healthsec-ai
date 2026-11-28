from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


class PercentileClipper:
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame) -> "PercentileClipper":
        for c in X.columns:
            lo, hi = np.nanpercentile(X[c].astype(float), [self.lower * 100, self.upper * 100])
            self.bounds_[c] = (float(lo), float(hi))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            if c in out.columns:
                out[c] = out[c].astype(float).clip(lo, hi)
        return out


class AnomalyGate:
    def __init__(self, contamination: float = 0.05, seed: int = 0):
        self.imputer = SimpleImputer(strategy="median")
        self.model = IsolationForest(
            contamination=contamination, random_state=seed, n_estimators=200
        )
        self.threshold_: float | None = None
        self.contamination = contamination

    def fit(self, X: pd.DataFrame) -> "AnomalyGate":
        Xt = self.imputer.fit_transform(X)
        self.model.fit(Xt)
        scores = -self.model.score_samples(Xt)
        self.threshold_ = float(np.quantile(scores, 1 - self.contamination))
        return self

    def scores(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self.imputer.transform(X))

    def keep_lowest(self, X: pd.DataFrame, frac: float = 0.8) -> np.ndarray:
        s = self.scores(X)
        k = max(1, int(round(frac * len(s))))
        keep = np.zeros(len(s), dtype=bool)
        keep[np.argsort(s)[:k]] = True
        return keep

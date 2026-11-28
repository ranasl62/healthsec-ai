from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseAdapter, TaskBundle

COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


class UCIHeartAdapter(BaseAdapter):
    name = "uci_heart"

    def load(self, root: Path) -> TaskBundle:
        path = root / "uci-heart" / "processed.cleveland.data"
        df = pd.read_csv(path, header=None, names=COLS, na_values="?")
        y = (df["num"].astype(float) > 0).astype(int)
        X = df.drop(columns=["num"]).astype(float)
        return TaskBundle(
            name=self.name,
            X=X,
            y=y,
            feature_names=list(X.columns),
            provenance={
                "source": "UCI Heart Disease (Cleveland processed)",
                "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
                "license": "CC BY 4.0",
                "n": int(len(df)),
                "events": int(y.sum()),
                "note": "Public demonstration task; not modern EHR cybersecurity proof",
            },
        )

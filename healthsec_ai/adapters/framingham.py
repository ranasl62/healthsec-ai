from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BaseAdapter, TaskBundle


class FraminghamAdapter(BaseAdapter):
    name = "framingham"

    def load(self, root: Path) -> TaskBundle:
        path = root / "framingham" / "framingham.csv"
        if not path.exists():
            path = root / "kaggle" / "framingham" / "framingham.csv"
        df = pd.read_csv(path)
        y = df["TenYearCHD"].astype(int)
        X = df.drop(columns=["TenYearCHD"]).astype(float)
        return TaskBundle(
            name=self.name,
            X=X,
            y=y,
            feature_names=list(X.columns),
            provenance={
                "source": "Kaggle: aasheesh200/framingham-heart-study-dataset",
                "url": "https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset",
                "n": int(len(df)),
                "events": int(y.sum()),
                "note": "Teaching Framingham CHD risk table; license Unknown on Kaggle — cite carefully",
            },
        )

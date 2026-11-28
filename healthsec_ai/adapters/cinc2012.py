from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseAdapter, TaskBundle

# PhysioNet Computing in Cardiology Challenge 2012 variables (subset)
SERIES = [
    "HR",
    "SysABP",
    "DiasABP",
    "MeanABP",
    "NISysABP",
    "NIDiasABP",
    "NIMeanABP",
    "RespRate",
    "Temp",
    "SpO2",
    "Glucose",
    "Creatinine",
    "BUN",
    "Na",
    "K",
    "HCO3",
    "HCT",
    "WBC",
    "Platelets",
    "Lactate",
    "Mg",
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "GCS",
    "Urine",
    "pH",
    "PaCO2",
    "PaO2",
    "FiO2",
]
STATIC = ["Age", "Gender", "Height", "Weight", "ICUType"]


def _parse_record(path: Path) -> dict:
    df = pd.read_csv(path)
    # columns: Time, Parameter, Value
    out: dict = {"RecordID": int(path.stem)}
    for p in STATIC:
        rows = df.loc[df["Parameter"] == p, "Value"]
        if not len(rows):
            out[p] = np.nan
            continue
        v = float(rows.iloc[0])
        # PhysioNet Challenge 2012 uses -1 as a missing sentinel for general descriptors.
        out[p] = np.nan if v == -1.0 else v
    for p in SERIES:
        vals = df.loc[df["Parameter"] == p, "Value"].astype(float)
        # Drop -1 sentinels if present in series (rare); keep clinical zeros.
        vals = vals[vals != -1.0]
        if len(vals) == 0:
            out[f"{p}_mean"] = np.nan
            out[f"{p}_min"] = np.nan
            out[f"{p}_max"] = np.nan
            out[f"{p}_last"] = np.nan
        else:
            out[f"{p}_mean"] = float(vals.mean())
            out[f"{p}_min"] = float(vals.min())
            out[f"{p}_max"] = float(vals.max())
            out[f"{p}_last"] = float(vals.iloc[-1])
    return out


class CinC2012Adapter(BaseAdapter):
    name = "cinc2012"

    def load(self, root: Path) -> TaskBundle:
        set_dir = root / "cinc2012" / "set-a"
        outcomes = pd.read_csv(root / "cinc2012" / "Outcomes-a.txt")
        # columns typically: RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death
        outcomes.columns = [c.strip() for c in outcomes.columns]
        death_col = [c for c in outcomes.columns if "death" in c.lower() or "Death" in c][0]
        id_col = [c for c in outcomes.columns if "record" in c.lower()][0]
        y_map = dict(zip(outcomes[id_col].astype(int), outcomes[death_col].astype(int)))

        rows = []
        for fp in sorted(set_dir.glob("*.txt")):
            rows.append(_parse_record(fp))
        data = pd.DataFrame(rows)
        data["y"] = data["RecordID"].map(y_map)
        data = data.dropna(subset=["y"])
        y = data["y"].astype(int)
        X = data.drop(columns=["RecordID", "y"])
        # ICUType is nominal (1-4); one-hot instead of feeding as a continuous number.
        if "ICUType" in X.columns:
            icu = X["ICUType"]
            X = X.drop(columns=["ICUType"])
            dummies = pd.get_dummies(icu.astype("Int64").astype(str), prefix="ICUType")
            # Drop dummy for missing ICUType if present
            dummies = dummies.drop(columns=[c for c in dummies.columns if c.endswith("nan") or c.endswith("<NA>")], errors="ignore")
            X = pd.concat([X, dummies.astype(float)], axis=1)
        X = X.astype(float)
        # Drop columns that are entirely missing (avoids imputer warnings)
        X = X.dropna(axis=1, how="all")
        return TaskBundle(
            name=self.name,
            X=X,
            y=y,
            feature_names=list(X.columns),
            provenance={
                "source": "PhysioNet Computing in Cardiology Challenge 2012 (set-a)",
                "url": "https://physionet.org/content/challenge-2012/",
                "n": int(len(X)),
                "events": int(y.sum()),
                "note": "Open PhysioNet challenge data; ICU mortality; first-48h aggregates; -1 sentinel->NaN; ICUType one-hot",
            },
        )

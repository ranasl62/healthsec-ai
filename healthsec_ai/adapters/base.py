from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TaskBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    provenance: dict


class BaseAdapter:
    name: str = "base"

    def load(self, root: Path) -> TaskBundle:
        raise NotImplementedError

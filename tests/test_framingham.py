from pathlib import Path

import pytest

from healthsec_ai.adapters import get_adapter

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_framingham_loads():
    path = ROOT / "framingham" / "framingham.csv"
    alt = ROOT / "kaggle" / "framingham" / "framingham.csv"
    if not path.exists() and not alt.exists():
        pytest.skip("Framingham CSV not present (data-dependent test)")
    b = get_adapter("framingham").load(ROOT)
    assert len(b.X) == 4240
    assert int(b.y.sum()) == 644
    assert "age" in b.feature_names

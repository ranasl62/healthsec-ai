from pathlib import Path

import pytest

from healthsec_ai.adapters import get_adapter, list_adapters

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_list_adapters():
    assert "uci_heart" in list_adapters()
    assert "cinc2012" in list_adapters()
    assert "synthetic_cv" in list_adapters()


def test_uci_loads():
    path = ROOT / "uci-heart" / "processed.cleveland.data"
    if not path.exists():
        pytest.skip("UCI Heart raw file not present (data-dependent test)")
    b = get_adapter("uci_heart").load(ROOT)
    assert len(b.X) >= 250
    assert b.y.nunique() == 2


def test_synthetic_loads():
    b = get_adapter("synthetic_cv").load(ROOT)
    assert len(b.X) == 4000

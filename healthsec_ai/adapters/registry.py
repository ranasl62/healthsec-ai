from __future__ import annotations

from .cinc2012 import CinC2012Adapter
from .framingham import FraminghamAdapter
from .synthetic import SyntheticCVAdapter
from .uci_heart import UCIHeartAdapter

_ADAPTERS = {
    UCIHeartAdapter.name: UCIHeartAdapter,
    CinC2012Adapter.name: CinC2012Adapter,
    FraminghamAdapter.name: FraminghamAdapter,
    SyntheticCVAdapter.name: SyntheticCVAdapter,
}


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)


def get_adapter(name: str):
    if name not in _ADAPTERS:
        raise KeyError(f"Unknown adapter '{name}'. Available: {list_adapters()}")
    return _ADAPTERS[name]()

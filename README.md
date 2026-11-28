# HealthSec-AI v0.2.1

Python toolkit for stress-testing tabular clinical prediction models under noise, missingness, and label flips. Not an LLM jailbreak suite and not an adaptive adversarial attack library.

**License:** MIT (`LICENSE`)  
**Python:** ≥3.10  
**Repository:** https://github.com/ranasl62/healthsec-ai  
**DOI:** https://doi.org/10.5281/zenodo.24579625  
**Entry points:** `healthsec` CLI **or** `python -m healthsec_ai.cli` (equivalent)

## Install

```bash
cd code
uv venv && source .venv/bin/activate
uv pip install -e .
# optional pinned stack used for reported tables:
uv pip install -r requirements-lock.txt
```

## Data (`data/raw/`) — not redistributed by default

| Adapter | How to obtain | Notes |
|---------|---------------|-------|
| `cinc2012` | PhysioNet Challenge 2012 | Open PhysioNet terms; cite Silva et al. |
| `framingham` | User-obtained teaching table / verified mirror | Do **not** assume Kaggle license is complete; verify before redistribution |
| `synthetic_cv` | Built-in generator | No PHI |
| `uci_heart` | UCI ML Repository | Smoke-test adapter only |

## Run (primary demos)

```bash
healthsec list-adapters
# equivalent: python -m healthsec_ai.cli list-adapters

healthsec run --dataset framingham --bootstrap 1000
healthsec run --dataset cinc2012 --bootstrap 1000
healthsec run --dataset synthetic_cv --bootstrap 1000
python scripts/make_figures.py
python scripts/make_architecture_figure.py
pytest -q
```

Outputs → `../results/` (`benchmark_*.csv`, `poison_multiseed_*.csv`, `risk_coverage_*.csv`, `*.meta.json`).

## Tests

`pytest` covers adapter listing, synthetic load, stress helpers, and metric invariants. Framingham/UCI loads skip when raw CSVs are absent. GitHub Actions (`.github/workflows/ci.yml`) runs the suite on Python 3.11/3.12.

## Citation

See `CITATION.cff`. Archive: https://doi.org/10.5281/zenodo.24579625

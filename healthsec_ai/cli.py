from __future__ import annotations

import argparse
from pathlib import Path

from healthsec_ai.adapters import list_adapters
from healthsec_ai.runner import run_benchmark


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="healthsec", description="HealthSec-AI benchmark CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-adapters", help="List dataset adapters")

    run_p = sub.add_parser("run", help="Run security benchmark on a dataset")
    run_p.add_argument("--dataset", required=True, choices=list_adapters())
    run_p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
    )
    run_p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results",
    )
    run_p.add_argument("--seed", type=int, default=20261126)
    run_p.add_argument("--bootstrap", type=int, default=1000)
    run_p.add_argument(
        "--models",
        default="logreg,rf,xgboost",
        help="Comma-separated model names",
    )

    args = p.parse_args(argv)
    if args.cmd == "list-adapters":
        print("\n".join(list_adapters()))
        return
    if args.cmd == "run":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        meta = run_benchmark(
            args.dataset,
            args.data_root,
            args.out,
            seed=args.seed,
            models=models,
            n_boot=args.bootstrap,
        )
        print(meta)


if __name__ == "__main__":
    main()

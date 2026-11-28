#!/usr/bin/env python3
"""Generate manuscript figures from verified v0.2 benchmark CSVs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG = ROOT / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Colorblind-safe Okabe–Ito-ish palette
C_LOGREG = "#0072B2"
C_RF = "#E69F00"
C_XGB = "#009E73"
C_SINGLE = "#0072B2"
C_NOISE = "#D55E00"
C_ABS = "#0072B2"


def load_all() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(RESULTS.glob("benchmark_*.csv"))]
    return pd.concat(frames, ignore_index=True)


def simplify_condition(c: str) -> str | None:
    if c == "clean":
        return "Clean"
    if c.startswith("atk_noise_eps"):
        return "Noise"
    if c == "atk_noise+clip":
        return "Noise+clip"
    if c.startswith("atk_miss"):
        return "Missingness"
    if c.startswith("atk_label_poison"):
        return "Label flip 5%"
    if c == "atk_noise+abstain80":
        return "Noise+abstain80"
    return None


def fig_auroc_primary_logreg():
    df = load_all()
    df = df[df["model"] == "logreg"].copy()
    df["cond"] = df["condition"].map(simplify_condition)
    df = df[df["cond"].notna()].drop_duplicates(subset=["dataset", "cond"], keep="first")
    order = ["Clean", "Noise", "Noise+clip", "Missingness", "Label flip 5%", "Noise+abstain80"]
    datasets = ["framingham", "cinc2012", "synthetic_cv"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].set_index("cond").reindex(order)
        y = sub["auroc"].to_numpy(dtype=float)
        lo = sub["auroc_ci_lo"].to_numpy(dtype=float)
        hi = sub["auroc_ci_hi"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        ax.bar(range(len(order)), y, color=C_SINGLE, yerr=yerr, capsize=3, ecolor="#222", width=0.75)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_title(ds, fontsize=11)
        ax.set_ylim(0.55, 0.95)
        ax.axhline(0.7, color="gray", lw=0.7, ls="--")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("AUROC (logistic regression)")
    fig.suptitle(
        "Primary tasks under simulated stresses (error bars = bootstrap 95% CI, n_boot=1000)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG / "Figure1_auroc_logreg.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(out)


def fig_cinc_three_models():
    df = load_all()
    df = df[df["dataset"] == "cinc2012"].copy()
    df["cond"] = df["condition"].map(simplify_condition)
    df = df[df["cond"].notna()].drop_duplicates(subset=["model", "cond"], keep="first")
    order = ["Clean", "Noise", "Missingness", "Label flip 5%"]
    models = ["logreg", "rf", "xgboost"]
    colors = [C_LOGREG, C_RF, C_XGB]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    width = 0.25
    x = np.arange(len(order))
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("cond").reindex(order)
        y = sub["auroc"].to_numpy(dtype=float)
        lo = sub["auroc_ci_lo"].to_numpy(dtype=float)
        hi = sub["auroc_ci_hi"].to_numpy(dtype=float)
        ax.bar(
            x + (i - 1) * width,
            y,
            width=width,
            label=model,
            color=colors[i],
            yerr=np.vstack([y - lo, hi - y]),
            capsize=2,
            ecolor="#222",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("AUROC")
    ax.set_title("CinC 2012: three models under simulated stresses (bootstrap 95% CI, n_boot=1000)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(0.65, 0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "Figure2_cinc2012_models.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(out)


def fig_brier_abstain():
    df = load_all()
    rows = []
    for ds in ["framingham", "cinc2012", "synthetic_cv"]:
        for model in ["logreg", "rf", "xgboost"]:
            sub = df[(df["dataset"] == ds) & (df["model"] == model)]
            absn = sub[sub["condition"] == "atk_noise+abstain80"]
            noise = sub[sub["condition"].str.startswith("atk_noise_eps")]
            if absn.empty or noise.empty:
                continue
            rows.append(
                {
                    "label": f"{ds}\n{model}",
                    "brier_noise": float(noise.iloc[0]["brier"]),
                    "brier_abstain": float(absn.iloc[0]["brier"]),
                }
            )
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["brier_noise"], width=0.4, label="Noise (cov=1.0)", color=C_NOISE)
    ax.bar(x + 0.2, t["brier_abstain"], width=0.4, label="Noise+abstain80", color=C_ABS)
    ax.set_xticks(list(x))
    ax.set_xticklabels(t["label"], fontsize=7)
    ax.set_ylabel("Brier score (lower better)")
    ax.set_title("Abstention under noise: Brier at 80% coverage (primary tasks)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "Figure3_abstention_brier.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(out)


def fig_risk_coverage_cinc():
    path = RESULTS / "risk_coverage_cinc2012.csv"
    if not path.exists():
        return
    rc = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, model in zip(axes, ["logreg", "rf", "xgboost"]):
        for setting, color, style in [
            ("clean", C_LOGREG, "-"),
            ("noise", C_NOISE, "--"),
        ]:
            sub = rc[(rc["model"] == model) & (rc["setting"] == setting)].sort_values("coverage")
            ax.plot(
                sub["coverage"],
                sub["brier"],
                style,
                marker="o",
                color=color,
                label=setting,
                linewidth=2,
            )
        ax.set_title(f"CinC · {model}")
        ax.set_xlabel("Coverage (fraction retained)")
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
    axes[0].set_ylabel("Brier (retained subset)")
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Risk-coverage under clean vs noise (Isolation Forest ranking)",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIG / "Figure4_risk_coverage_cinc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(out)


def fig_poison_multiseed():
    frames = [pd.read_csv(p) for p in sorted(RESULTS.glob("poison_multiseed_*.csv"))]
    if not frames:
        return
    t = pd.concat(frames, ignore_index=True)
    t = t[t["dataset"].isin(["framingham", "cinc2012", "synthetic_cv"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    datasets = ["framingham", "cinc2012", "synthetic_cv"]
    models = ["logreg", "rf", "xgboost"]
    colors = [C_LOGREG, C_RF, C_XGB]
    width = 0.25
    x = np.arange(len(datasets))
    for i, model in enumerate(models):
        means, sds = [], []
        for ds in datasets:
            row = t[(t["dataset"] == ds) & (t["model"] == model)]
            means.append(float(row.iloc[0]["auroc_mean"]) if len(row) else np.nan)
            sds.append(float(row.iloc[0]["auroc_sd"]) if len(row) else 0.0)
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            yerr=sds,
            capsize=3,
            label=model,
            color=colors[i],
            ecolor="#222",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("AUROC mean ± SD (5 flip seeds)")
    ax.set_title("5% random label flips across five seeds (primary tasks)")
    ax.legend(frameon=False)
    ax.set_ylim(0.6, 0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "Figure5_poison_multiseed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(out)


def fig_calibration_cinc():
    """Reliability-style calibration summary: slope/intercept bars for CinC primary conditions."""
    path = RESULTS / "benchmark_cinc2012.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = ["clean", "atk_noise_eps0.5", "atk_miss"]
    rows = []
    for _, r in df.iterrows():
        cond = r["condition"]
        if cond == "clean":
            label = "Clean"
        elif cond.startswith("atk_noise_eps"):
            label = "Noise"
        elif cond.startswith("atk_miss"):
            label = "Missingness"
        else:
            continue
        rows.append(
            {
                "model": r["model"],
                "label": label,
                "slope": r["calib_slope"],
                "intercept": r["calib_intercept"],
            }
        )
    t = pd.DataFrame(rows)
    order = ["Clean", "Noise", "Missingness"]
    models = ["logreg", "rf", "xgboost"]
    colors = [C_LOGREG, C_RF, C_XGB]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(order))
    width = 0.25
    for ax, metric, ylab in [
        (axes[0], "slope", "Calibration slope (ideal ≈ 1)"),
        (axes[1], "intercept", "Calibration intercept (ideal ≈ 0)"),
    ]:
        for i, model in enumerate(models):
            vals = []
            for lab in order:
                sub = t[(t["model"] == model) & (t["label"] == lab)]
                vals.append(float(sub.iloc[0][metric]) if len(sub) else np.nan)
            ax.bar(x + (i - 1) * width, vals, width=width, label=model, color=colors[i])
        ax.set_xticks(list(x))
        ax.set_xticklabels(order)
        ax.set_ylabel(ylab)
        ax.axhline(1.0 if metric == "slope" else 0.0, color="gray", ls="--", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("CinC 2012 calibration slope and intercept under clean / noise / missingness", fontsize=11)
    fig.tight_layout()
    out = FIG / "Figure7_calibration_cinc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(out)


if __name__ == "__main__":
    fig_auroc_primary_logreg()
    fig_cinc_three_models()
    fig_brier_abstain()
    fig_risk_coverage_cinc()
    fig_poison_multiseed()
    fig_calibration_cinc()

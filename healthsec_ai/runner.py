from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthsec_ai.adapters import get_adapter
from healthsec_ai.attacks.engine import (
    feature_noise,
    fgsm_feature_shift,
    label_poison,
    label_poison_targeted,
    missingness,
)
from healthsec_ai.defenses.engine import AnomalyGate, PercentileClipper
from healthsec_ai.evaluation.metrics import (
    bootstrap_metric_ci,
    paired_bootstrap_delta_ci,
    risk_coverage_curve,
    summarize,
)

POISON_SEEDS = [11, 22, 33, 44, 55]


def _model(name: str, seed: int):
    if name in ("logreg", "logreg_unweighted"):
        cw = "balanced" if name == "logreg" else None
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight=cw, random_state=seed
                    ),
                ),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(name)


def _importance_topk(clf, feature_names: list[str], k: int = 3) -> list[str]:
    """Train-fold importance only (prespecified protocol)."""
    est = clf.named_steps["clf"]
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_)
    else:
        imp = np.abs(np.asarray(est.coef_).ravel())
    order = np.argsort(-imp)
    return [feature_names[i] for i in order[: min(k, len(feature_names))]]


def _row(dataset, model, condition, y, p, seed, n_boot, **extra):
    m = summarize(y, p)
    a_m, a_lo, a_hi = bootstrap_metric_ci(y, p, "auroc", n_boot, seed)
    b_m, b_lo, b_hi = bootstrap_metric_ci(y, p, "brier", n_boot, seed + 1)
    return {
        "dataset": dataset,
        "model": model,
        "condition": condition,
        **m,
        "auroc_boot_mean": a_m,
        "auroc_ci_lo": a_lo,
        "auroc_ci_hi": a_hi,
        "brier_boot_mean": b_m,
        "brier_ci_lo": b_lo,
        "brier_ci_hi": b_hi,
        "attacks_simulated": True,
        **extra,
    }


def run_benchmark(
    dataset: str,
    data_root: Path,
    out_dir: Path,
    seed: int = 20261126,
    models: list[str] | None = None,
    n_boot: int = 1000,
) -> dict:
    models = models or ["logreg", "rf", "xgboost"]
    adapter = get_adapter(dataset)
    bundle = adapter.load(data_root)
    X, y = bundle.X, bundle.y
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    yte_np = yte.to_numpy()
    ytr_np = ytr.to_numpy()
    feat_names = list(X.columns)

    rows = []
    poison_summary_rows = []
    poison_per_seed_rows = []
    coverage_rows = []
    delta_rows = []
    anomaly_shift_rows = []
    # For cross-model difference-of-deltas (noise harm)
    noise_preds: dict[str, np.ndarray] = {}
    clean_preds: dict[str, np.ndarray] = {}

    for mname in models:
        clf = _model(mname, seed)
        clf.fit(Xtr, ytr)
        top = _importance_topk(clf, feat_names, k=3)

        p_clean = clf.predict_proba(Xte)[:, 1]
        clean_preds[mname] = p_clean
        rows.append(_row(dataset, mname, "clean", yte_np, p_clean, seed, n_boot))

        X_noise = feature_noise(Xte, eps=0.5, seed=seed, scale_ref=Xtr)
        p_noise = clf.predict_proba(X_noise)[:, 1]
        noise_preds[mname] = p_noise
        rows.append(_row(dataset, mname, "atk_noise_eps0.5", yte_np, p_noise, seed, n_boot))

        clip = PercentileClipper().fit(Xtr)
        p_clip = clf.predict_proba(clip.transform(X_noise))[:, 1]
        rows.append(_row(dataset, mname, "atk_noise+clip", yte_np, p_clip, seed, n_boot))

        # Closed-form FGSM-style and targeted flip baselines (linear models only)
        if mname in ("logreg", "logreg_unweighted"):
            coef = np.asarray(clf.named_steps["clf"].coef_).ravel()
            X_fgsm = fgsm_feature_shift(Xte, coef, Xtr, eps=0.5, direction="increase_risk")
            p_fgsm = clf.predict_proba(X_fgsm)[:, 1]
            rows.append(_row(dataset, mname, "atk_fgsm_eps0.5", yte_np, p_fgsm, seed, n_boot))
            d_fgsm = paired_bootstrap_delta_ci(
                yte_np, p_clean, p_fgsm, metric="auroc", n_boot=n_boot, seed=seed + 19
            )
            delta_rows.append(
                {
                    "dataset": dataset,
                    "model": mname,
                    "contrast": "fgsm_minus_clean",
                    "auroc_clean": float(summarize(yte_np, p_clean)["auroc"]),
                    "auroc_alt": float(summarize(yte_np, p_fgsm)["auroc"]),
                    "delta_auroc": d_fgsm["delta"],
                    "delta_ci_lo": d_fgsm["ci_lo"],
                    "delta_ci_hi": d_fgsm["ci_hi"],
                    "n_boot": n_boot,
                }
            )
            p_tr = clf.predict_proba(Xtr)[:, 1]
            y_tgt = label_poison_targeted(ytr_np, p_tr, 0.05, seed=POISON_SEEDS[0])
            clf_t = _model(mname, seed)
            clf_t.fit(Xtr, y_tgt)
            p_tgt = clf_t.predict_proba(Xte)[:, 1]
            rows.append(
                _row(dataset, mname, "atk_label_targeted_0.05_seed11", yte_np, p_tgt, seed, n_boot)
            )

        X_miss = missingness(Xte, top)
        p_miss = clf.predict_proba(X_miss)[:, 1]
        rows.append(
            _row(
                dataset,
                mname,
                f"atk_miss_trainImp:{','.join(top)}",
                yte_np,
                p_miss,
                seed,
                n_boot,
                dropped_features=",".join(top),
            )
        )

        for label, p_base, p_alt in [
            ("noise_minus_clean", p_clean, p_noise),
            ("missingness_minus_clean", p_clean, p_miss),
            ("clip_minus_noise", p_noise, p_clip),
            ("clip_minus_clean", p_clean, p_clip),
        ]:
            d = paired_bootstrap_delta_ci(
                yte_np, p_base, p_alt, metric="auroc", n_boot=n_boot, seed=seed + 17
            )
            delta_rows.append(
                {
                    "dataset": dataset,
                    "model": mname,
                    "contrast": label,
                    "auroc_clean": float(summarize(yte_np, p_base)["auroc"]),
                    "auroc_alt": float(summarize(yte_np, p_alt)["auroc"]),
                    "delta_auroc": d["delta"],
                    "delta_ci_lo": d["ci_lo"],
                    "delta_ci_hi": d["ci_hi"],
                    "n_boot": n_boot,
                }
            )

        # Multi-seed random label flips
        poison_metrics = []
        n_train = len(ytr_np)
        n_flip = int(round(0.05 * n_train))
        for ps in POISON_SEEDS:
            y_pois = label_poison(ytr_np, 0.05, seed=ps)
            clf_p = _model(mname, seed)
            clf_p.fit(Xtr, y_pois)
            p_p = clf_p.predict_proba(Xte)[:, 1]
            m = summarize(yte_np, p_p)
            poison_metrics.append(m)
            poison_per_seed_rows.append(
                {
                    "dataset": dataset,
                    "model": mname,
                    "poison_seed": ps,
                    "n_train": n_train,
                    "n_flips": n_flip,
                    "train_event_rate_clean": float(ytr_np.mean()),
                    "train_event_rate_flipped": float(y_pois.mean()),
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "brier": m["brier"],
                }
            )
        poison_summary_rows.append(
            {
                "dataset": dataset,
                "model": mname,
                "attack": "label_flip_0.05_random",
                "n_seeds": len(poison_metrics),
                "n_flips_per_seed": n_flip,
                "auroc_mean": float(np.mean([m["auroc"] for m in poison_metrics])),
                "auroc_sd": float(np.std([m["auroc"] for m in poison_metrics], ddof=1)),
                "auprc_mean": float(np.mean([m["auprc"] for m in poison_metrics])),
                "brier_mean": float(np.mean([m["brier"] for m in poison_metrics])),
                "clean_auroc": float(summarize(yte_np, p_clean)["auroc"]),
            }
        )
        y_pois0 = label_poison(ytr_np, 0.05, seed=POISON_SEEDS[0])
        clf_p0 = _model(mname, seed)
        clf_p0.fit(Xtr, y_pois0)
        rows.append(
            _row(
                dataset,
                mname,
                "atk_label_poison_0.05_seed11",
                yte_np,
                clf_p0.predict_proba(Xte)[:, 1],
                seed,
                n_boot,
            )
        )

        gate = AnomalyGate(seed=seed).fit(Xtr)
        s_clean = gate.scores(Xte)
        s_noise = gate.scores(X_noise)
        anomaly_shift_rows.append(
            {
                "dataset": dataset,
                "model": mname,
                "score_mean_clean": float(np.mean(s_clean)),
                "score_mean_noise": float(np.mean(s_noise)),
                "score_mean_delta": float(np.mean(s_noise) - np.mean(s_clean)),
                "score_median_clean": float(np.median(s_clean)),
                "score_median_noise": float(np.median(s_noise)),
                "note": "IF fit on clean train; scores not assumed exchangeable across clean vs noise",
            }
        )
        # Ranking scores for selective prediction baselines
        conf_noise = np.maximum(p_noise, 1.0 - p_noise)
        conf_clean = np.maximum(p_clean, 1.0 - p_clean)
        rng_abs = np.random.default_rng(seed + 101)
        rand_noise = rng_abs.random(len(yte_np))
        rand_clean = np.random.default_rng(seed + 102).random(len(yte_np))

        for setting, probs, scores, keep_mode, gate_name in [
            ("noise", p_noise, s_noise, "lowest", "anomaly"),
            ("clean", p_clean, s_clean, "lowest", "anomaly"),
            ("noise", p_noise, conf_noise, "highest", "confidence"),
            ("clean", p_clean, conf_clean, "highest", "confidence"),
            ("noise", p_noise, rand_noise, "lowest", "random"),
            ("clean", p_clean, rand_clean, "lowest", "random"),
        ]:
            for rc in risk_coverage_curve(yte_np, probs, scores, keep=keep_mode):
                coverage_rows.append(
                    {
                        "dataset": dataset,
                        "model": mname,
                        "setting": setting,
                        "gate": gate_name,
                        **rc,
                    }
                )

        keep = gate.keep_lowest(X_noise, frac=0.8)
        events_total = float(yte_np.sum())
        frac_ev_abs = (
            float((events_total - float(yte_np[keep].sum())) / events_total)
            if events_total > 0
            else float("nan")
        )
        rows.append(
            _row(
                dataset,
                mname,
                "atk_noise+abstain80",
                yte_np[keep],
                p_noise[keep],
                seed,
                n_boot,
                coverage=float(keep.mean()),
                event_rate_full=float(yte_np.mean()),
                frac_events_abstained=frac_ev_abs,
                abstention_gate="anomaly_if",
            )
        )
        # Confidence and random abstention baselines at 80% coverage (noisy inputs)
        for gate_name, scores, keep_mode in [
            ("confidence", conf_noise, "highest"),
            ("random", rand_noise, "lowest"),
        ]:
            order = np.argsort(scores)
            if keep_mode == "highest":
                order = order[::-1]
            k = max(1, int(round(0.8 * len(yte_np))))
            idx = order[:k]
            keep_b = np.zeros(len(yte_np), dtype=bool)
            keep_b[idx] = True
            frac_b = (
                float((events_total - float(yte_np[keep_b].sum())) / events_total)
                if events_total > 0
                else float("nan")
            )
            rows.append(
                _row(
                    dataset,
                    mname,
                    f"atk_noise+abstain80_{gate_name}",
                    yte_np[keep_b],
                    p_noise[keep_b],
                    seed,
                    n_boot,
                    coverage=0.8,
                    event_rate_full=float(yte_np.mean()),
                    frac_events_abstained=frac_b,
                    abstention_gate=gate_name,
                )
            )

    # Difference-of-deltas: (noise_AUROC - clean_AUROC)_modelA vs modelB on same boots
    cross_delta_rows = []
    primary_models = [m for m in models if m in clean_preds and m in noise_preds]
    for i, ma in enumerate(primary_models):
        for mb in primary_models[i + 1 :]:
            # delta_m = AUROC(noise) - AUROC(clean); compare delta_b - delta_a
            rng = np.random.default_rng(seed + 91)
            n = len(yte_np)
            diffs = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                yt = yte_np[idx]
                if yt.min() == yt.max():
                    continue
                da = roc_auc_score(yt, noise_preds[ma][idx]) - roc_auc_score(
                    yt, clean_preds[ma][idx]
                )
                db = roc_auc_score(yt, noise_preds[mb][idx]) - roc_auc_score(
                    yt, clean_preds[mb][idx]
                )
                diffs.append(db - da)
            point = (
                float(summarize(yte_np, noise_preds[mb])["auroc"] - summarize(yte_np, clean_preds[mb])["auroc"])
                - float(summarize(yte_np, noise_preds[ma])["auroc"] - summarize(yte_np, clean_preds[ma])["auroc"])
            )
            arr = np.asarray(diffs) if diffs else np.array([np.nan])
            cross_delta_rows.append(
                {
                    "dataset": dataset,
                    "model_a": ma,
                    "model_b": mb,
                    "contrast": "noise_delta_b_minus_noise_delta_a",
                    "delta": point,
                    "delta_ci_lo": float(np.nanpercentile(arr, 2.5)),
                    "delta_ci_hi": float(np.nanpercentile(arr, 97.5)),
                    "n_boot": n_boot,
                    "note": "delta=(noise-clean)_b-(noise-clean)_a; positive => model_a loses more AUROC under noise",
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table_path = out_dir / f"benchmark_{dataset}.csv"
    table.to_csv(table_path, index=False)
    pd.DataFrame(poison_summary_rows).to_csv(
        out_dir / f"poison_multiseed_{dataset}.csv", index=False
    )
    pd.DataFrame(poison_per_seed_rows).to_csv(
        out_dir / f"poison_per_seed_{dataset}.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(out_dir / f"risk_coverage_{dataset}.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out_dir / f"paired_deltas_{dataset}.csv", index=False)
    pd.DataFrame(anomaly_shift_rows).to_csv(
        out_dir / f"anomaly_score_shift_{dataset}.csv", index=False
    )
    pd.DataFrame(cross_delta_rows).to_csv(
        out_dir / f"cross_model_noise_deltas_{dataset}.csv", index=False
    )
    meta = {
        "dataset": dataset,
        "provenance": bundle.provenance,
        "seed": seed,
        "n_boot": n_boot,
        "models": models,
        "poison_seeds": POISON_SEEDS,
        "importance_protocol": "train-fold coefficient/impurity importance only",
        "table": str(table_path),
        "attacks_simulated": True,
        "framework": "healthsec-ai==0.2.1",
        "role": "primary" if dataset != "uci_heart" else "smoke_test_only",
        "inference": "paired_bootstrap_deltas_plus_cross_model_noise_deltas",
        "package_path_note": "stress modules live under attacks/ for historical naming; stresses are non-adaptive unless fgsm/targeted rows",
    }
    (out_dir / f"benchmark_{dataset}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta

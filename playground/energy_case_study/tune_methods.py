"""Search sklearn / LightGBM method configurations for the energy/oil case study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from config import CONFIG_PATH, CaseStudyConfig, ModelConfig, load_config
from experiment import _run_window, build_candidates, output_dir

from data import build_energy_case_study_service


def parse_args() -> argparse.Namespace:
    """Parse method-search CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run a larger grid (includes log_return and more lag variants; slower).",
    )
    parser.add_argument(
        "--include-slow-models",
        action="store_true",
        help="Also search slower random-forest variants.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class CandidateSpec:
    """One tuning candidate."""

    model_id: str
    label: str
    estimator: str
    covariate_group: str | None
    lags: int
    covariate_lags: int | None
    horizon: int
    target_strategy: str
    feature_mode: str
    alpha: float = 1.0
    n_estimators: int = 200


def _candidate_specs(*, full: bool = False, include_slow_models: bool = False) -> list[CandidateSpec]:
    """Return a compact grid across horizons, targets, and estimators."""
    specs: list[CandidateSpec] = []
    horizons = [10, 20, 30]

    for horizon in horizons:
        specs.append(
            CandidateSpec(
                model_id=f"h{horizon}_ridge_uni_level_raw",
                label=f"{horizon}b ridge univariate level (raw)",
                estimator="ridge",
                covariate_group=None,
                lags=5,
                covariate_lags=None,
                horizon=horizon,
                target_strategy="level",
                feature_mode="raw_levels",
            )
        )

        for group, gid in [("no_futures", "nf"), ("with_futures", "wf")]:
            for est, eid in [("huber", "huber"), ("lightgbm", "lgbm")]:
                specs.append(
                    CandidateSpec(
                        model_id=f"h{horizon}_{eid}_{gid}_d10_pd_eng",
                        label=f"{horizon}b {eid} {group} price_delta engineered 10/5",
                        estimator=est,
                        covariate_group=group,
                        lags=10,
                        covariate_lags=5,
                        horizon=horizon,
                        target_strategy="price_delta",
                        feature_mode="engineered",
                    )
                )

            specs.append(
                CandidateSpec(
                    model_id=f"h{horizon}_ridge_{gid}_d10_pd_eng",
                    label=f"{horizon}b ridge {group} price_delta engineered 10/5",
                    estimator="ridge",
                    covariate_group=group,
                    lags=10,
                    covariate_lags=5,
                    horizon=horizon,
                    target_strategy="price_delta",
                    feature_mode="engineered",
                )
            )

        if full:
            specs.append(
                CandidateSpec(
                    model_id=f"h{horizon}_lgbm_wf_lr_d10_eng",
                    label=f"{horizon}b lightgbm with_futures log_return engineered 10/5",
                    estimator="lightgbm",
                    covariate_group="with_futures",
                    lags=10,
                    covariate_lags=5,
                    horizon=horizon,
                    target_strategy="log_return",
                    feature_mode="engineered",
                )
            )
            for lags, cov_lags in [(5, 3), (20, 5)]:
                specs.append(
                    CandidateSpec(
                        model_id=f"h{horizon}_huber_wf_l{lags}_pd",
                        label=f"{horizon}b huber with_futures price_delta {lags}/{cov_lags}",
                        estimator="huber",
                        covariate_group="with_futures",
                        lags=lags,
                        covariate_lags=cov_lags,
                        horizon=horizon,
                        target_strategy="price_delta",
                        feature_mode="engineered",
                    )
                )

    if include_slow_models:
        for horizon in horizons:
            specs.append(
                CandidateSpec(
                    model_id=f"h{horizon}_rf_uni_level",
                    label=f"{horizon}b random_forest univariate level",
                    estimator="random_forest",
                    covariate_group=None,
                    lags=10,
                    covariate_lags=None,
                    horizon=horizon,
                    target_strategy="level",
                    feature_mode="raw_levels",
                )
            )

    return specs


def _config_for_candidate(base: CaseStudyConfig, spec: CandidateSpec, *, output_subdir: str) -> CaseStudyConfig:
    """Create a single-candidate config for tuning."""
    model = ModelConfig(
        label=spec.label,
        method="sklearn_residual",
        estimator=spec.estimator,  # type: ignore[arg-type]
        covariate_group=spec.covariate_group,
        lags=spec.lags,
        lags_past_covariates=spec.covariate_lags,
        n_estimators=spec.n_estimators,
        alpha=spec.alpha,
        target_strategy=spec.target_strategy,  # type: ignore[arg-type]
        feature_mode=spec.feature_mode,  # type: ignore[arg-type]
    )
    return base.model_copy(
        update={
            "forecast": base.forecast.model_copy(update={"default_horizon": spec.horizon}),
            "models": {spec.model_id: model},
            "artifacts": base.artifacts.model_copy(update={"output_dir": output_dir(base) / output_subdir}),
        }
    )


def run_search(
    config_path: Path = CONFIG_PATH,
    *,
    full: bool = False,
    include_slow_models: bool = False,
) -> Path:
    """Run the tuning grid and write ranked metrics."""
    base = load_config(config_path)
    service = build_energy_case_study_service(base)
    rows: list[dict[str, object]] = []

    for spec in _candidate_specs(full=full, include_slow_models=include_slow_models):
        config = _config_for_candidate(base, spec, output_subdir="method_search")
        candidates = build_candidates(config)
        model_selection_predictions, model_selection_metrics = _run_window(
            config=config,
            data_service=service,
            candidates=candidates,
            window="model_selection",
        )
        q1_predictions, q1_metrics = _run_window(
            config=config,
            data_service=service,
            candidates=candidates,
            window="q1_rollforward",
        )
        del model_selection_predictions, q1_predictions
        rows.append(
            {
                "model_id": spec.model_id,
                "label": spec.label,
                "horizon": spec.horizon,
                "estimator": spec.estimator,
                "covariate_group": spec.covariate_group or "univariate",
                "target_strategy": spec.target_strategy,
                "feature_mode": spec.feature_mode,
                "lags": spec.lags,
                "covariate_lags": spec.covariate_lags,
                "model_selection_mean_crps": float(model_selection_metrics.iloc[0]["mean_crps"]),
                "model_selection_mae": float(model_selection_metrics.iloc[0]["mae"]),
                "q1_mean_crps": float(q1_metrics.iloc[0]["mean_crps"]),
                "q1_mae": float(q1_metrics.iloc[0]["mae"]),
                "q1_alarm_rate": float(q1_metrics.iloc[0]["alarm_rate"]),
            }
        )

    results = pd.DataFrame(rows).sort_values(["model_selection_mean_crps", "q1_mean_crps"])
    path = output_dir(base) / "method_search" / "method_search_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    print(results.head(16).to_string(index=False))
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    args = parse_args()
    run_search(args.config, full=args.full, include_slow_models=args.include_slow_models)

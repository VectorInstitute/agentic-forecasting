"""Unit tests for ``food_price_forecasting.analysis``.

These tests construct synthetic :class:`BacktestResult` objects with known
predictions and call the analysis helpers directly — no network fetches, no
real models.  The goal is to pin the tidy-DataFrame shape produced by
:func:`predictions_to_dataframe` and the exact CFPR average-over-average YoY
semantics of :func:`compute_avgyoy` against hand-computed values.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from food_price_forecasting.analysis import (
    compute_avgyoy,
    predictions_to_dataframe,
    rationales_table,
    summarize_crps,
)


CFPR_HORIZONS: list[int] = list(range(6, 18))  # Jan..Dec of Y+1 from a July origin.


def _make_task(task_id: str = "cpi_food_overall", target_series_id: str = "cpi_food_canada") -> ForecastingTask:
    return ForecastingTask(
        task_id=task_id,
        target_series_id=target_series_id,
        horizons=CFPR_HORIZONS,
        frequency="MS",
        description="CFPR trajectory test task.",
    )


def _make_prediction(
    *,
    predictor_id: str,
    task_id: str,
    origin: datetime,
    horizon: int,
    point: float,
    spread: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> Prediction:
    forecast_date = (pd.Timestamp(origin) + pd.DateOffset(months=horizon)).to_pydatetime()
    quantiles = {
        0.05: point - 2 * spread,
        0.20: point - spread,
        0.50: point,
        0.80: point + spread,
        0.95: point + 2 * spread,
    }
    return Prediction(
        predictor_id=predictor_id,
        task_id=task_id,
        issued_at=origin,
        as_of=origin,
        forecast_date=forecast_date,
        payload=ContinuousForecast(point_forecast=point, quantiles=quantiles),
        metadata=metadata or {},
    )


def _make_backtest_result(
    *,
    predictor_id: str,
    task: ForecastingTask,
    origins: list[datetime],
    point_trajectory: dict[int, float],
    spread: float = 1.0,
    metadata_by_horizon: dict[int, dict[str, object]] | None = None,
) -> BacktestResult:
    # BacktestSpec requires ``start < end``; pad the end by one day so the
    # single-origin fixture case remains valid.
    end = (pd.Timestamp(origins[-1]) + pd.Timedelta(days=1)).to_pydatetime()
    spec = BacktestSpec(
        task=task,
        start=origins[0],
        end=end,
        stride=12,
        warmup=0,
        description="synthetic",
    )
    predictions: list[Prediction] = []
    scores: list[float] = []
    for origin in origins:
        for h, pt in point_trajectory.items():
            pred = _make_prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                origin=origin,
                horizon=h,
                point=pt,
                spread=spread,
                metadata=(metadata_by_horizon or {}).get(h),
            )
            predictions.append(pred)
            scores.append(0.5 * h)
    return BacktestResult(
        spec=spec,
        predictor_id=predictor_id,
        predictions=predictions,
        scores=scores,
        mean_crps=float(np.mean(scores)),
        ran_at=datetime(2025, 1, 1),
        skipped_origins=0,
    )


class TestPredictionsToDataframe:
    def test_single_backtest_result_flattens_to_tidy_df(self) -> None:
        task = _make_task()
        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2022, 7, 1), datetime(2023, 7, 1)],
            point_trajectory={h: 160.0 + h for h in CFPR_HORIZONS},
        )

        df = predictions_to_dataframe(result)

        expected_cols = {
            "predictor_id",
            "task_id",
            "origin",
            "origin_year",
            "horizon",
            "forecast_date",
            "median",
            "crps",
        }
        assert expected_cols.issubset(df.columns)
        assert len(df) == 2 * len(CFPR_HORIZONS)
        assert set(df["predictor_id"].unique()) == {"naive_last"}
        assert set(df["task_id"].unique()) == {task.task_id}
        assert set(df["horizon"].unique()) == set(CFPR_HORIZONS)
        assert set(df["origin_year"].unique()) == {2022, 2023}

    def test_dict_of_results_uses_keys_as_task_ids(self) -> None:
        task_a = _make_task(task_id="cpi_food_overall", target_series_id="cpi_food_canada")
        task_b = _make_task(task_id="cpi_meat", target_series_id="cpi_meat_canada")
        results: dict[str, BacktestResult] = {
            "cpi_food_overall": _make_backtest_result(
                predictor_id="naive_last",
                task=task_a,
                origins=[datetime(2023, 7, 1)],
                point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
            ),
            "cpi_meat": _make_backtest_result(
                predictor_id="naive_last",
                task=task_b,
                origins=[datetime(2023, 7, 1)],
                point_trajectory=dict.fromkeys(CFPR_HORIZONS, 170.0),
            ),
        }

        df = predictions_to_dataframe(results)

        assert set(df["task_id"].unique()) == {"cpi_food_overall", "cpi_meat"}
        assert (df.loc[df["task_id"] == "cpi_meat", "median"] == 170.0).all()
        assert (df.loc[df["task_id"] == "cpi_food_overall", "median"] == 160.0).all()

    def test_overrides_predictor_id(self) -> None:
        task = _make_task()
        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2023, 7, 1)],
            point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
        )

        df = predictions_to_dataframe(result, predictor_id="custom_id")

        assert set(df["predictor_id"].unique()) == {"custom_id"}

    def test_horizon_matches_months_between_origin_and_forecast_date(self) -> None:
        task = _make_task()
        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2023, 7, 1)],
            point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
        )

        df = predictions_to_dataframe(result).sort_values("horizon").reset_index(drop=True)

        for row in df.itertuples():
            origin = pd.Timestamp(row.origin)
            fd = pd.Timestamp(row.forecast_date)
            months = (fd.year - origin.year) * 12 + (fd.month - origin.month)
            assert int(row.horizon) == months


class TestComputeAvgYoY:
    @staticmethod
    def _monthly_actuals(years: list[int], value: float = 160.0) -> pd.DataFrame:
        timestamps: list[pd.Timestamp] = []
        values: list[float] = []
        for y in years:
            for m in range(1, 13):
                timestamps.append(pd.Timestamp(y, m, 1))
                values.append(value)
        return pd.DataFrame({"timestamp": timestamps, "value": values})

    def test_flat_series_and_flat_forecast_yields_zero_yoy(self) -> None:
        task = _make_task()
        actual_df = self._monthly_actuals([2022, 2023], value=160.0)

        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2022, 7, 1)],
            point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
        )

        yoy = compute_avgyoy(result, actual_df)

        assert list(yoy.columns) == [
            "origin_year",
            "actual_avg_y0",
            "predicted_avg_y1",
            "yoy_median",
            "yoy_q05",
            "yoy_q25",
            "yoy_q75",
            "yoy_q95",
            "actual_yoy",
        ]
        assert len(yoy) == 1
        row = yoy.iloc[0]
        assert row["origin_year"] == 2022
        assert row["actual_avg_y0"] == 160.0
        assert row["predicted_avg_y1"] == 160.0
        assert row["yoy_median"] == 0.0
        assert row["actual_yoy"] == 0.0

    def test_predicted_avg_matches_mean_of_trajectory(self) -> None:
        task = _make_task()
        actual_df = self._monthly_actuals([2022], value=160.0)

        traj = {h: 160.0 + h for h in CFPR_HORIZONS}
        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2022, 7, 1)],
            point_trajectory=traj,
        )

        yoy = compute_avgyoy(result, actual_df)

        row = yoy.iloc[0]
        expected_pred_avg = float(np.mean(list(traj.values())))
        assert row["predicted_avg_y1"] == expected_pred_avg
        assert row["yoy_median"] == (expected_pred_avg / 160.0 - 1)
        assert pd.isna(row["actual_yoy"])

    def test_quantile_ordering_preserved_on_symmetric_spread(self) -> None:
        task = _make_task()
        actual_df = self._monthly_actuals([2022], value=160.0)

        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2022, 7, 1)],
            point_trajectory=dict.fromkeys(CFPR_HORIZONS, 170.0),
            spread=2.0,
        )

        yoy = compute_avgyoy(result, actual_df).iloc[0]
        assert yoy["yoy_q05"] < yoy["yoy_q25"] < yoy["yoy_median"] < yoy["yoy_q75"] < yoy["yoy_q95"]
        assert abs((yoy["yoy_median"] - yoy["yoy_q25"]) - (yoy["yoy_q75"] - yoy["yoy_median"])) < 1e-12

    def test_origin_skipped_when_year_y_incomplete(self) -> None:
        task = _make_task()
        # Only 6 months of 2022 actuals - should be skipped.
        partial = self._monthly_actuals([2022], value=160.0).iloc[:6].copy()

        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2022, 7, 1)],
            point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
        )

        yoy = compute_avgyoy(result, partial)
        assert yoy.empty

    def test_actual_yoy_realised_when_year_y1_complete(self) -> None:
        task = _make_task()
        rows: list[dict[str, object]] = []
        for m in range(1, 13):
            rows.append({"timestamp": pd.Timestamp(2022, m, 1), "value": 160.0})
            rows.append({"timestamp": pd.Timestamp(2023, m, 1), "value": 168.0})
        actual_df = pd.DataFrame(rows)

        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2022, 7, 1)],
            point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
        )

        yoy = compute_avgyoy(result, actual_df).iloc[0]
        assert yoy["actual_yoy"] == (168.0 / 160.0 - 1)

    def test_multi_origin_sorted_output(self) -> None:
        task = _make_task()
        actual_df = self._monthly_actuals([2021, 2022, 2023, 2024], value=160.0)

        result = _make_backtest_result(
            predictor_id="naive_last",
            task=task,
            origins=[datetime(2021, 7, 1), datetime(2022, 7, 1), datetime(2023, 7, 1)],
            point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
        )

        yoy = compute_avgyoy(result, actual_df)

        assert list(yoy["origin_year"]) == [2021, 2022, 2023]


class TestSummarizeCRPS:
    def test_leaderboard_shape_and_mean_row(self) -> None:
        task_a = _make_task(task_id="cpi_food_overall")
        task_b = _make_task(task_id="cpi_meat")

        results_by_predictor: dict[str, dict[str, BacktestResult]] = {
            "naive": {
                "cpi_food_overall": _make_backtest_result(
                    predictor_id="naive",
                    task=task_a,
                    origins=[datetime(2023, 7, 1)],
                    point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
                ),
                "cpi_meat": _make_backtest_result(
                    predictor_id="naive",
                    task=task_b,
                    origins=[datetime(2023, 7, 1)],
                    point_trajectory=dict.fromkeys(CFPR_HORIZONS, 170.0),
                ),
            },
        }

        board = summarize_crps(results_by_predictor)

        assert "naive" in board.columns
        assert set(board.index) - {"MEAN"} == {"cpi_food_overall", "cpi_meat"}
        assert (
            board.loc["MEAN", "naive"]
            == (
                results_by_predictor["naive"]["cpi_food_overall"].mean_crps
                + results_by_predictor["naive"]["cpi_meat"].mean_crps
            )
            / 2
        )


class TestRationalesTable:
    def test_metadata_keys_become_columns(self) -> None:
        task = _make_task()
        meta = {6: {"rationale": "flat signal"}, 7: {"rationale": "slight uptick"}}
        result = _make_backtest_result(
            predictor_id="llm_demo",
            task=task,
            origins=[datetime(2023, 7, 1)],
            point_trajectory=dict.fromkeys(CFPR_HORIZONS, 160.0),
            metadata_by_horizon=meta,
        )

        df = rationales_table(result)
        assert "meta_rationale" in df.columns
        with_rat = df.dropna(subset=["meta_rationale"])
        assert sorted(with_rat["horizon"].tolist()) == [6, 7]

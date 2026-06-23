"""Analysis helpers for the BoC rate-decision experiment.

Pure functions that turn :class:`BacktestResult` / :class:`EvalResult`
objects into tidy DataFrames for binary and ordered-categorical evaluations:
per-meeting prediction tables, score leaderboards, reliability/calibration
bins, and rationale extracts.

Kept separate from the notebooks so they can be unit-tested and reused.
All functions are pure: they take results plus an observed event series and
return DataFrames. They never fetch data or mutate global state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.eval import EvalResult
from aieng.forecasting.evaluation.prediction import BinaryForecast, CategoricalForecast
from aieng.forecasting.evaluation.task import ForecastingTask


def _task_from_result(result: BacktestResult | EvalResult) -> ForecastingTask:
    """Return the forecasting task attached to a backtest or eval result."""
    if isinstance(result, BacktestResult):
        return result.spec.task
    return result.eval_spec.task


def predictions_to_frame(
    results: dict[str, BacktestResult | EvalResult],
    event_df: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten binary/categorical predictions into a tidy per-meeting DataFrame.

    Parameters
    ----------
    results : dict[str, BacktestResult | EvalResult]
        Mapping ``predictor_id -> result``. Binary results contribute
        ``probability`` rows; categorical results contribute one ``p_<label>``
        column per task-declared category.
    event_df : pd.DataFrame
        Observed event series (``timestamp`` / ``value`` columns, as returned
        by :meth:`DataService.get_series`), used to attach the realised outcome
        at each prediction's ``forecast_date``. Binary series use 0/1 values;
        direction series use the task category values (for example -1/0/+1).

    Returns
    -------
    pd.DataFrame
        Columns: ``predictor_id``, ``origin``, ``meeting_date``, ``score``,
        ``metric``, ``outcome``, ``probability`` for binary rows, and one
        ``p_<label>`` column per categorical label. Categorical rows also carry
        ``outcome_label`` so one-vs-rest views can be derived without the task
        object.
    """
    outcome_by_date = {
        pd.Timestamp(ts).normalize(): float(v) for ts, v in zip(event_df["timestamp"], event_df["value"], strict=True)
    }

    rows: list[dict[str, object]] = []
    for predictor_id, result in results.items():
        task = _task_from_result(result)
        value_to_label = {category.value: category.label for category in task.categories or []}
        category_labels = [category.label for category in task.categories or []]
        for pred, score in zip(result.predictions, result.scores, strict=True):
            meeting_date = pd.Timestamp(pred.forecast_date).normalize()
            outcome = outcome_by_date.get(meeting_date)
            row: dict[str, object] = {
                "predictor_id": predictor_id,
                "origin": pd.Timestamp(pred.as_of),
                "meeting_date": meeting_date,
                "score": float(score),
                "metric": result.metric,
                "outcome": outcome,
                "probability": np.nan,
            }
            if isinstance(pred.payload, BinaryForecast):
                row["probability"] = float(pred.payload.probability)
                row["outcome_label"] = None
            elif isinstance(pred.payload, CategoricalForecast):
                row["outcome_label"] = value_to_label.get(outcome) if outcome is not None else None
                for label in category_labels:
                    row[f"p_{label}"] = float(pred.payload.probabilities[label])
            else:
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def score_leaderboard(
    results: dict[str, BacktestResult | EvalResult],
    *,
    reference_id: str | None = None,
) -> pd.DataFrame:
    """Build a mean-score leaderboard, optionally with skill scores.

    Skill against a reference predictor is ``1 - score / score_ref``: positive
    means the predictor beats the reference, 0 means it matches it, negative
    means it loses. The reference is usually a historical-frequency baseline
    for binary tasks or a categorical-frequency baseline for direction tasks.

    Parameters
    ----------
    results : dict[str, BacktestResult | EvalResult]
        Mapping ``predictor_id -> result``. Lower scores are better.
    reference_id : str or None
        Predictor id to use as the skill-score reference. When ``None`` (or
        not present in ``results``) the ``skill_vs_reference`` column is
        omitted. If the reference mean score is non-positive, the skill column
        is also omitted.

    Returns
    -------
    pd.DataFrame
        One row per predictor, sorted by ``mean_score`` ascending. Columns:
        ``predictor_id``, ``metric``, ``mean_score``, ``n_predictions``,
        ``n_skipped_origins`` and optionally ``skill_vs_reference``.
    """
    rows: list[dict[str, object]] = []
    for predictor_id, result in results.items():
        rows.append(
            {
                "predictor_id": predictor_id,
                "metric": result.metric,
                "mean_score": result.mean_score,
                "n_predictions": len(result.predictions),
                "n_skipped_origins": result.skipped_origins,
            }
        )
    board = pd.DataFrame(rows).sort_values("mean_score").reset_index(drop=True)

    if reference_id is not None and reference_id in results:
        reference_score = results[reference_id].mean_score
        if reference_score > 0:
            board["skill_vs_reference"] = (1.0 - board["mean_score"] / reference_score).round(4)
    return board


def one_vs_rest_frame(predictions_df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Convert a categorical tidy frame into a binary one-vs-rest frame.

    The input must come from :func:`predictions_to_frame` for a categorical
    task, which provides ``p_<label>`` probability columns and an
    ``outcome_label`` column derived from the task category values. For
    example, ``one_vs_rest_frame(df, "cut")`` returns ``probability = p_cut``
    and ``outcome = 1`` for realised cuts, ``0`` for realised holds/hikes, and
    ``NaN`` when the meeting has not resolved.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Categorical tidy frame from :func:`predictions_to_frame`.
    category : str
        Category label to evaluate one-vs-rest.

    Returns
    -------
    pd.DataFrame
        Columns: ``predictor_id``, ``meeting_date``, ``probability``,
        ``outcome``.
    """
    probability_col = f"p_{category}"
    required = {"predictor_id", "meeting_date", "outcome_label", probability_col}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df is missing required categorical columns: {sorted(missing)}")

    frame = predictions_df.loc[:, ["predictor_id", "meeting_date", probability_col, "outcome_label"]].copy()
    frame = frame.rename(columns={probability_col: "probability"})
    frame["outcome"] = np.where(
        frame["outcome_label"].isna(),
        np.nan,
        (frame["outcome_label"] == category).astype(float),
    )
    return frame.loc[:, ["predictor_id", "meeting_date", "probability", "outcome"]]


def calibration_table(
    predictions_df: pd.DataFrame,
    *,
    predictor_id: str | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Bin predicted probabilities and compare against observed event frequency.

    This is the tabular form of the reliability curve: a perfectly calibrated
    predictor has ``observed_frequency ~= mean_predicted`` in every bin.
    Pass binary-task frames directly, or pass the output of
    :func:`one_vs_rest_frame` for a categorical category such as ``cut`` or
    ``hike``. With only ~120 meetings, bins are necessarily coarse — five
    equal-width bins is about as fine as the sample supports.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Binary-style frame with ``probability`` and 0/1 ``outcome`` columns.
        Rows with missing ``probability`` or ``outcome`` are dropped.
    predictor_id : str or None
        Restrict to one predictor; ``None`` uses all rows (caller's
        responsibility to pass a single-predictor frame in that case).
    n_bins : int
        Number of equal-width probability bins over [0, 1].

    Returns
    -------
    pd.DataFrame
        One row per non-empty bin: ``bin_left``, ``bin_right``,
        ``mean_predicted``, ``observed_frequency``, ``n``.
    """
    df = predictions_df.dropna(subset=["probability", "outcome"])
    if predictor_id is not None:
        df = df[df["predictor_id"] == predictor_id]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        # Right-inclusive last bin so p=1.0 is counted.
        upper_ok = df["probability"] <= right if right >= 1.0 else df["probability"] < right
        in_bin = df[(df["probability"] >= left) & upper_ok]
        if in_bin.empty:
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "mean_predicted": float(in_bin["probability"].mean()),
                "observed_frequency": float(in_bin["outcome"].mean()),
                "n": int(len(in_bin)),
            }
        )
    return pd.DataFrame(rows)


def yearly_outcome_table(event_df: pd.DataFrame, labels: dict[float, str] | None = None) -> pd.DataFrame:
    """Summarise meeting outcomes per calendar year.

    Parameters
    ----------
    event_df : pd.DataFrame
        Observed event series (``timestamp`` / ``value`` columns).
    labels : dict[float, str] or None
        Optional mapping from observed category value to display label. When
        omitted, preserves the binary cut-event summary.

    Returns
    -------
    pd.DataFrame
        Indexed by year. Binary mode returns ``n_meetings``, ``n_cuts``, and
        ``cut_rate``. Categorical mode returns ``n_meetings`` plus one
        ``n_<label>`` column per supplied label.
    """
    df = event_df.copy()
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year
    if labels is not None:
        grouped = df.groupby("year")["value"].agg(n_meetings="count")
        for value, label in labels.items():
            grouped[f"n_{label}"] = df["value"].eq(value).groupby(df["year"]).sum().astype(int)
        return grouped

    grouped = df.groupby("year")["value"].agg(n_meetings="count", n_cuts="sum")
    grouped["n_cuts"] = grouped["n_cuts"].astype(int)
    grouped["cut_rate"] = (grouped["n_cuts"] / grouped["n_meetings"]).round(3)
    return grouped


def rationales_table(result: BacktestResult | EvalResult) -> pd.DataFrame:
    """Extract per-prediction metadata (reasoning traces etc.) into a DataFrame.

    For the agent predictor, ``metadata`` carries ``reasoning`` and
    ``key_signals`` — the inputs for the planned reasoning-alignment
    evaluation against the Bank's own published rationale.

    Parameters
    ----------
    result : BacktestResult | EvalResult
        Result to introspect.

    Returns
    -------
    pd.DataFrame
        Columns: ``origin``, ``meeting_date``, ``probability`` for binary
        payloads or one ``p_<label>`` column per categorical label, plus one
        ``meta_*`` column per distinct metadata key (missing values filled
        with ``None``).
    """
    task = _task_from_result(result)
    category_labels = [category.label for category in task.categories or []]
    base_rows: list[dict[str, object]] = []
    all_keys: set[str] = set()
    for pred in result.predictions:
        row: dict[str, object] = {
            "origin": pd.Timestamp(pred.as_of),
            "meeting_date": pd.Timestamp(pred.forecast_date),
        }
        if isinstance(pred.payload, BinaryForecast):
            row["probability"] = float(pred.payload.probability)
        elif isinstance(pred.payload, CategoricalForecast):
            for label in category_labels:
                row[f"p_{label}"] = float(pred.payload.probabilities[label])
        for k, v in pred.metadata.items():
            row[f"meta_{k}"] = v
            all_keys.add(f"meta_{k}")
        base_rows.append(row)

    for row in base_rows:
        for k in all_keys:
            row.setdefault(k, None)
    return pd.DataFrame(base_rows)


@dataclass(frozen=True)
class PanelRow:
    """One method's prediction for a single meeting, for the decision panel.

    Attributes
    ----------
    predictor_id : str
        Identifier of the predictor that produced this row.
    probabilities : dict[str, float]
        Predicted probability per category label, in task-category order.
    score : float
        The meeting's score for this predictor (RPS for the 3-way task).
    rationale : str
        Stated reasoning, if the method recorded one (agents and LLMPs do via
        ``Prediction.metadata["rationale"]``); empty string otherwise.
    key_signals : list[str]
        Supporting signals, if recorded (agents only today); empty otherwise.
    """

    predictor_id: str
    probabilities: dict[str, float]
    score: float
    rationale: str = ""
    key_signals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionPanel:
    """Everything needed to render one meeting's decision panel across methods.

    Attributes
    ----------
    meeting_date : pd.Timestamp
        The announcement date being predicted.
    origin : pd.Timestamp
        Forecast origin (``as_of``) the predictions were issued from.
    categories : list[str]
        Ordered category labels (e.g. ``["cut", "hold", "hike"]``).
    outcome_label : str or None
        Realised decision at ``meeting_date``, or ``None`` if unresolved.
    prior_outcome_label : str or None
        The most recent resolved decision strictly before ``meeting_date``,
        for context; ``None`` if none is available.
    rows : list[PanelRow]
        One row per predictor, in the order the results were supplied.
    """

    meeting_date: pd.Timestamp
    origin: pd.Timestamp
    categories: list[str]
    outcome_label: str | None
    prior_outcome_label: str | None
    rows: list[PanelRow]


def decision_panel_data(
    results: dict[str, BacktestResult | EvalResult],
    event_df: pd.DataFrame,
    *,
    meeting_date: str | pd.Timestamp | None = None,
) -> DecisionPanel:
    """Assemble one meeting's cross-method prediction panel.

    Gathers, for a single announcement date, each categorical method's
    predicted distribution, its score, and any stated ``rationale`` /
    ``key_signals`` (read from ``Prediction.metadata`` exactly as
    :func:`rationales_table` does), plus the realised outcome and the prior
    decision for context.

    Parameters
    ----------
    results : dict[str, BacktestResult | EvalResult]
        Mapping ``predictor_id -> result``. Only categorical results
        contribute; binary-only inputs raise ``ValueError``.
    event_df : pd.DataFrame
        Observed direction series (``timestamp`` / ``value``), used for the
        realised and prior outcomes (values mapped to labels via the task's
        declared categories).
    meeting_date : str | pd.Timestamp | None
        Which announcement to assemble. ``None`` (default) selects the most
        recent meeting present across the categorical results.

    Returns
    -------
    DecisionPanel
    """
    categorical = [(pid, r) for pid, r in results.items() if _task_from_result(r).payload_type == "categorical"]
    if not categorical:
        raise ValueError("decision_panel_data requires at least one categorical result.")

    task = _task_from_result(categorical[0][1])
    categories = [category.label for category in task.categories or []]
    value_to_label = {category.value: category.label for category in task.categories or []}
    outcome_by_date = {
        pd.Timestamp(ts).normalize(): float(v) for ts, v in zip(event_df["timestamp"], event_df["value"], strict=True)
    }

    all_dates = sorted(
        {pd.Timestamp(pred.forecast_date).normalize() for _, result in categorical for pred in result.predictions}
    )
    if not all_dates:
        raise ValueError("No categorical predictions found in results.")
    target = all_dates[-1] if meeting_date is None else pd.Timestamp(meeting_date).normalize()

    rows: list[PanelRow] = []
    origin: pd.Timestamp | None = None
    for predictor_id, result in categorical:
        for pred, score in zip(result.predictions, result.scores, strict=True):
            if pd.Timestamp(pred.forecast_date).normalize() != target:
                continue
            if not isinstance(pred.payload, CategoricalForecast):
                continue
            metadata = pred.metadata or {}
            rationale = str(metadata.get("rationale", "") or "").strip()
            key_signals = list(metadata.get("key_signals", []) or [])
            rows.append(
                PanelRow(
                    predictor_id=predictor_id,
                    probabilities={label: float(pred.payload.probabilities[label]) for label in categories},
                    score=float(score),
                    rationale=rationale,
                    key_signals=key_signals,
                )
            )
            origin = pd.Timestamp(pred.as_of)
            break

    outcome_value = outcome_by_date.get(target)
    outcome_label = value_to_label.get(outcome_value) if outcome_value is not None else None

    prior_dates = sorted(d for d in outcome_by_date if d < target)
    prior_outcome_label = value_to_label.get(outcome_by_date[prior_dates[-1]]) if prior_dates else None

    return DecisionPanel(
        meeting_date=target,
        origin=origin if origin is not None else target,
        categories=categories,
        outcome_label=outcome_label,
        prior_outcome_label=prior_outcome_label,
        rows=rows,
    )


def panel_rationales_markdown(panel: DecisionPanel, labels: dict[str, str] | None = None) -> str:
    """Render a decision panel's rationales as a Markdown string.

    One block per method that recorded a ``rationale`` and/or ``key_signals``;
    methods without either are skipped. Intended for ``IPython.display.Markdown``
    in notebooks, keeping rationale prose out of the matplotlib figure.

    Parameters
    ----------
    panel : DecisionPanel
        Assembled by :func:`decision_panel_data`.
    labels : dict[str, str] or None
        Optional predictor_id -> display-label map for the block headings.

    Returns
    -------
    str
        Markdown text, or the empty string when no method recorded a rationale.
    """
    label_map = {row.predictor_id: (labels or {}).get(row.predictor_id, row.predictor_id) for row in panel.rows}
    blocks: list[str] = []
    for row in panel.rows:
        if not row.rationale and not row.key_signals:
            continue
        block = f"**{label_map[row.predictor_id]}**"
        if row.key_signals:
            block += "\n\nKey signals: " + ", ".join(row.key_signals)
        if row.rationale:
            block += f"\n\n{row.rationale}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


__all__ = [
    "DecisionPanel",
    "PanelRow",
    "calibration_table",
    "decision_panel_data",
    "one_vs_rest_frame",
    "panel_rationales_markdown",
    "predictions_to_frame",
    "rationales_table",
    "score_leaderboard",
    "yearly_outcome_table",
]

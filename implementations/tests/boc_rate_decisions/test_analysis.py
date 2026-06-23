"""Unit tests for ``boc_rate_decisions.analysis.decision_panel_data``.

Builds small synthetic categorical :class:`BacktestResult`s with known
cut/hold/hike distributions, metadata, and outcomes, then checks the assembled
:class:`DecisionPanel` against hand-set values — no network, no models.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec
from aieng.forecasting.evaluation.prediction import CategoricalForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory
from boc_rate_decisions.analysis import decision_panel_data, panel_rationales_markdown


_CATEGORIES = [
    TaskCategory(label="cut", value=-1.0),
    TaskCategory(label="hold", value=0.0),
    TaskCategory(label="hike", value=1.0),
]

# Two announcement dates and a prior resolved meeting for context.
_PRIOR = pd.Timestamp("2024-03-06")
_MEETINGS = [pd.Timestamp("2024-06-05"), pd.Timestamp("2024-09-04")]


def _task() -> ForecastingTask:
    return ForecastingTask(
        task_id="boc_rate_direction_next_meeting",
        target_series_id="boc_rate_decision_direction",
        horizons=[28],
        frequency="D",
        description="3-way direction test task.",
        payload_type="categorical",
        categories=_CATEGORIES,
    )


def _result(
    *,
    predictor_id: str,
    dists: dict[pd.Timestamp, dict[str, float]],
    scores: dict[pd.Timestamp, float],
    metadata: dict[pd.Timestamp, dict[str, object]] | None = None,
) -> BacktestResult:
    task = _task()
    preds: list[Prediction] = []
    score_list: list[float] = []
    for meeting in _MEETINGS:
        origin = (meeting - pd.Timedelta(days=28)).to_pydatetime()
        preds.append(
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=origin,
                as_of=origin,
                forecast_date=meeting.to_pydatetime(),
                payload=CategoricalForecast(probabilities=dists[meeting]),
                metadata=(metadata or {}).get(meeting, {}),
            )
        )
        score_list.append(scores[meeting])
    return BacktestResult(
        spec=BacktestSpec(
            task=task,
            start=_MEETINGS[0].to_pydatetime(),
            end=(_MEETINGS[-1] + pd.Timedelta(days=1)).to_pydatetime(),
            stride=1,
            warmup=0,
            description="synthetic",
        ),
        predictor_id=predictor_id,
        predictions=preds,
        scores=score_list,
        metric="rps",
        mean_score=float(sum(score_list) / len(score_list)),
        ran_at=datetime(2025, 1, 1),
        skipped_origins=0,
    )


def _event_df() -> pd.DataFrame:
    # prior hold, then two cuts.
    return pd.DataFrame({"timestamp": [_PRIOR, _MEETINGS[0], _MEETINGS[1]], "value": [0.0, -1.0, -1.0]})


def _results() -> dict[str, BacktestResult]:
    return {
        "agent": _result(
            predictor_id="agent",
            dists={
                _MEETINGS[0]: {"cut": 0.45, "hold": 0.50, "hike": 0.05},
                _MEETINGS[1]: {"cut": 0.80, "hold": 0.18, "hike": 0.02},
            },
            scores={_MEETINGS[0]: 0.30, _MEETINGS[1]: 0.05},
            metadata={
                _MEETINGS[1]: {"rationale": "Easing cycle underway.", "key_signals": ["yield -1.23", "rate momentum"]},
            },
        ),
        "climatology": _result(
            predictor_id="climatology",
            dists={
                _MEETINGS[0]: {"cut": 0.05, "hold": 0.80, "hike": 0.15},
                _MEETINGS[1]: {"cut": 0.06, "hold": 0.79, "hike": 0.15},
            },
            scores={_MEETINGS[0]: 0.62, _MEETINGS[1]: 0.61},
        ),
    }


def test_defaults_to_most_recent_meeting() -> None:
    panel = decision_panel_data(_results(), _event_df())
    assert panel.meeting_date == _MEETINGS[1]
    assert panel.origin == _MEETINGS[1] - pd.Timedelta(days=28)
    assert panel.categories == ["cut", "hold", "hike"]
    assert panel.outcome_label == "cut"
    assert panel.prior_outcome_label == "cut"  # the 2024-06-05 cut precedes 2024-09-04
    assert [row.predictor_id for row in panel.rows] == ["agent", "climatology"]


def test_explicit_meeting_and_row_contents() -> None:
    panel = decision_panel_data(_results(), _event_df(), meeting_date="2024-06-05")
    assert panel.meeting_date == _MEETINGS[0]
    assert panel.outcome_label == "cut"
    assert panel.prior_outcome_label == "hold"

    by_id = {row.predictor_id: row for row in panel.rows}
    assert by_id["agent"].probabilities == {"cut": 0.45, "hold": 0.50, "hike": 0.05}
    assert by_id["agent"].score == 0.30
    # No metadata for the June meeting → empty rationale / signals.
    assert by_id["agent"].rationale == ""
    assert by_id["agent"].key_signals == []


def test_rationale_and_signals_are_extracted() -> None:
    panel = decision_panel_data(_results(), _event_df())  # most recent (September)
    agent = next(row for row in panel.rows if row.predictor_id == "agent")
    assert agent.rationale == "Easing cycle underway."
    assert agent.key_signals == ["yield -1.23", "rate momentum"]
    climatology = next(row for row in panel.rows if row.predictor_id == "climatology")
    assert climatology.rationale == ""


def test_requires_categorical_results() -> None:
    with pytest.raises(ValueError, match="categorical"):
        decision_panel_data({}, _event_df())


def test_panel_rationales_markdown_includes_only_methods_with_rationale() -> None:
    panel = decision_panel_data(_results(), _event_df())  # September: agent has rationale + signals
    md = panel_rationales_markdown(panel, {"agent": "Agent", "climatology": "Climatology"})
    assert "**Agent**" in md
    assert "Easing cycle underway." in md
    assert "Key signals: yield -1.23, rate momentum" in md
    assert "Climatology" not in md  # no rationale → block skipped


def test_panel_rationales_markdown_empty_when_no_rationales() -> None:
    panel = decision_panel_data(_results(), _event_df(), meeting_date="2024-06-05")  # no metadata
    assert panel_rationales_markdown(panel) == ""

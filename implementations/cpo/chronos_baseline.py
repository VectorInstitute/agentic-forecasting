r"""Chronos zero-shot foundation-model baseline for the weekly MPOB price.

Chronos (Amazon) is a time series foundation model: it is not trained on this
series at all, only on a large public corpus of other series, and forecasts
ours purely from the shape of the numbers it is shown at inference time --
"zero-shot". That is a genuinely different hypothesis from every other
predictor in this lineup, which are all fit fresh at each cutoff.

Why leakage is the central risk here
-------------------------------------
A model fit fresh at each cutoff (ETS, ARIMA, Prophet, ...) cannot leak,
because it is trained only on the history visible at that cutoff. A
foundation model is different: its weights were fixed once, long before we
ever call it, on a pretraining corpus whose exact contents Amazon does not
publish in full. If that corpus happened to include MPOB prices (or a highly
correlated series) covering the weeks we are asking it to forecast, the model
could already know the answer -- not because it is a good forecaster, but
because it memorised it.

The fix follows directly from how these models are built: training data
collection finishes, then the model is trained, evaluated, and only then
released. So a checkpoint's public release date is always *later* than its
training data cutoff, never earlier or equal to it. That makes release date a
safe (if conservative) stand-in for "the last date this model could possibly
have seen": if a cutoff falls on or before the checkpoint's release date, the
model provably cannot have seen the real prices we are asking it to forecast,
because they did not exist yet when its training data was collected.

``predict()`` enforces this automatically: it returns ``[]`` (no leaderboard
row at all) for any cutoff before ``release_date``, so an unsafe
predictor/cutoff pairing fails silent-and-empty rather than silently leaking.
See ``README.md`` / ``03_baselines.ipynb`` for the full per-checkpoint,
per-cutoff safety table this is built from.

Why pin ``revision``, not just ``checkpoint``
-----------------------------------------------
Hugging Face model repos are not static after release. Checked via the HF
API on 2026-08-14, every checkpoint below has a ``lastModified`` well after
its ``createdAt`` -- e.g. ``amazon/chronos-2`` was created 2025-10-30 but
last touched 2026-06-05, eight months later. Loading a checkpoint by name
alone (``from_pretrained("amazon/chronos-2")``) resolves to whatever is on
the ``main`` branch *today*, which may not be the same weights that existed
on the release date we are trusting. Pinning an explicit commit ``revision``
makes the run reproducible regardless of what happens to the repo later, and
is the only way to be sure we are testing the checkpoint the release date
actually refers to.

Verified checkpoints (HF API, checked 2026-08-14)
----------------------------------------------------
======================  ==========  ================  ==========================================
checkpoint              release*    revision (pinned)  safe cutoffs (>= release, cpo_cutoffs.yaml)
======================  ==========  ================  ==========================================
amazon/chronos-t5-small 2024-02-21  a971ba2194...      2024-05-03 onward (6 of 7)
amazon/chronos-bolt-base 2024-11-25 5d9f166d69...      2024-11-29 onward (4 of 7)
amazon/chronos-2        2025-10-30  29ec3766d3...      2025-11-28 only  (1 of 7)
======================  ==========  ================  ==========================================
\\* HF ``createdAt`` for the repo, the earliest publicly-checkable date -- more
conservative than the paper/announcement date, which is what we want for a
leakage boundary. Full revision hashes are in ``03_baselines.ipynb``.

Interface differences between checkpoints
--------------------------------------------
``BaseChronosPipeline.from_pretrained`` auto-dispatches to the right
architecture (the original T5 sampling model, Chronos-Bolt's direct quantile
head, or Chronos-2's encoder), and all three expose ``predict_quantiles``
with the same signature -- except Chronos-2, which is multivariate-capable
and so expects a 3-D ``(n_series, n_variates, history_length)`` input and
returns a list of per-series tensors rather than a single batched tensor.
Confirmed empirically (2026-08-14): passing a 1-D tensor to Chronos-2 raises
``ValueError``, not a silent shape error, so the branch below is exercised
and validated, not merely defensive.

Uncertainty
-----------
Unlike Prophet's Gaussian-from-interval-width conversion, Chronos accepts
:data:`aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES` directly
as ``quantile_levels`` for the original and Chronos-2 pipelines, which both
return genuinely distinct values across the full 11-level grid (checked
empirically 2026-08-14: no two levels coincide).

Chronos-Bolt is the exception. It has a quantile-regression head trained on
exactly nine fixed levels, ``[0.1, ..., 0.9]``, and asking it for 0.05/0.95
does not extrapolate -- confirmed empirically (2026-08-14) that it silently
returns the *exact same* value as 0.1/0.9 (a zero-width sliver at each tail),
alongside a library warning that this "may significantly affect the quality
of the predictions." Silently accepting that would understate Bolt's tail
uncertainty in every backtest it runs. Detected via ``isinstance(...,
ChronosBoltPipeline)`` and handled the same way as
:mod:`cpo.timesfm_baseline`'s missing tails: request the native nine levels,
then linearly extrapolate 0.05 and 0.95 from the two nearest deciles.

Segfault alongside LightGBM
----------------------------
Running this predictor in the same process as ``darts_lightgbm`` (as the
notebook does -- everything runs in one kernel) crashes with SIGSEGV, no
Python traceback, right after the checkpoint's weights finish loading.
Isolated empirically (2026-08-17) to multi-threaded torch contending with
LightGBM's own OpenMP runtime -- the same family of native-library conflict
as the ``libomp.dylib`` crash fixed earlier in this project, just triggered
by a different pair of libraries. ``KMP_DUPLICATE_LIB_OK=TRUE`` alone does
*not* fix it (tested); ``torch.set_num_threads(1)`` alone does (tested), so
that is the fix applied below, not the environment variable workaround.
LightGBM's own predictor is already pinned to ``num_threads=1`` for the same
reason (see ``03_baselines.ipynb``); this is the torch side of that same
constraint.

**Prerequisite:** ``chronos-forecasting`` (a project dependency; pulls torch,
transformers).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.evaluation.task import ForecastingTask

_MIN_OBSERVATIONS = 10

# Chronos-Bolt's trained quantile-head levels -- see module docstring.
_BOLT_NATIVE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _extrapolate_tail(levels: list[float], values: list[float], target: float) -> float:
    """Linearly extrapolate ``target`` from the two nearest native quantiles."""
    if target < levels[0]:
        x0, x1, y0, y1 = levels[0], levels[1], values[0], values[1]
    else:
        x0, x1, y0, y1 = levels[-2], levels[-1], values[-2], values[-1]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (target - x0)


class WeeklyChronosPredictor(Predictor):
    """Zero-shot Chronos forecast, gated against pretraining leakage.

    Parameters
    ----------
    checkpoint : str
        Hugging Face repo id, e.g. ``"amazon/chronos-t5-small"``.
    revision : str
        Pinned commit sha -- not a branch name. See the module docstring for
        why: repos change after release, and a name alone is not
        reproducible.
    release_date : str
        The checkpoint's earliest publicly-checkable date (HF ``createdAt``).
        ``predict()`` returns ``[]`` for any cutoff strictly before this
        date -- see the module docstring for why that is the correct and
        sufficient leakage guard.
    device_map : str
        Passed through to ``from_pretrained``. ``"cpu"`` by default since
        these are single-series, weekly-cadence calls, not a training loop.
    """

    def __init__(
        self,
        checkpoint: str,
        revision: str,
        release_date: str,
        device_map: str = "cpu",
    ) -> None:
        self._checkpoint = checkpoint
        self._revision = revision
        self._release_date = pd.Timestamp(release_date)
        self._device_map = device_map
        self._pipeline: Any = None

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier derived from the checkpoint name."""
        name = self._checkpoint.split("/")[-1].replace("-", "_")
        return name if name.startswith("chronos_") else f"chronos_{name}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Forecast every horizon in one zero-shot call, or skip if unsafe.

        Parameters
        ----------
        task : ForecastingTask
            Target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.

        Returns
        -------
        list[Prediction]
            One :class:`ContinuousForecast` per horizon; empty if
            ``context.as_of`` predates ``release_date`` (leakage guard) or
            visible history is shorter than :data:`_MIN_OBSERVATIONS`.
        """
        if pd.Timestamp(context.as_of) < self._release_date:
            return []

        series = context.get_series(task.target_series_id).set_index("timestamp")["value"]
        if len(series) < _MIN_OBSERVATIONS:
            return []

        import torch  # noqa: PLC0415
        from chronos import BaseChronosPipeline, Chronos2Pipeline, ChronosBoltPipeline  # noqa: PLC0415

        torch.set_num_threads(1)  # see module docstring: avoids a segfault alongside LightGBM in-process

        if self._pipeline is None:
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self._checkpoint, revision=self._revision, device_map=self._device_map
            )

        values = torch.tensor(series.to_numpy(dtype="float32"))
        is_chronos2 = isinstance(self._pipeline, Chronos2Pipeline)
        is_bolt = isinstance(self._pipeline, ChronosBoltPipeline)
        model_input = values.reshape(1, 1, -1) if is_chronos2 else values

        request_levels = _BOLT_NATIVE_LEVELS if is_bolt else STANDARD_QUANTILES
        quantiles, mean = self._pipeline.predict_quantiles(
            model_input, prediction_length=task.horizon, quantile_levels=request_levels
        )
        if is_chronos2:
            quantiles, mean = quantiles[0][0], mean[0][0]
        else:
            quantiles, mean = quantiles[0], mean[0]

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []
        for h in task.horizons:
            step = h - 1
            native_values = [float(v) for v in quantiles[step]]
            if is_bolt:
                row_quantiles = dict(zip(_BOLT_NATIVE_LEVELS, native_values, strict=True))
                for q in STANDARD_QUANTILES:
                    if q not in row_quantiles:
                        row_quantiles[q] = _extrapolate_tail(_BOLT_NATIVE_LEVELS, native_values, q)
            else:
                row_quantiles = dict(zip(request_levels, native_values, strict=True))
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=float(mean[step]),
                        quantiles={q: float(row_quantiles[q]) for q in STANDARD_QUANTILES},
                    ),
                )
            )
        return predictions


__all__ = ["WeeklyChronosPredictor"]

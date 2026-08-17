r"""TimesFM zero-shot foundation-model baseline for the weekly MPOB price.

TimesFM (Google) is the second time series foundation model in this lineup,
alongside :mod:`cpo.chronos_baseline`. Same hypothesis, different pretraining
corpus and architecture -- see that module's docstring for why a zero-shot
foundation model is worth testing at all, and for the leakage reasoning this
module reuses without repeating.

Only the 2.5 checkpoint is available here
--------------------------------------------
The ``timesfm`` PyPI package was rewritten for the 2.5 release: as installed
(version 2.0.2, checked 2026-08-14) it exposes exactly one loadable class,
``TimesFM_2p5_200M_torch``, and no code path for the older 1.0 or 2.0
checkpoints -- confirmed by inspecting the package's public API directly, not
assumed from changelogs. Getting 1.0 or 2.0 running would mean installing the
``timesfm==1.0.0`` release from 2024, which predates this stack (likely a
different, jax-based dependency chain) and was not worth the fragility for
the extra cutoffs it would unlock. Practically this means TimesFM is
leak-safe for exactly one of the seven cutoffs -- see the table below.

Verified checkpoint (HF API, checked 2026-08-14)
-----------------------------------------------------
====================================  ==========  ==================  ==========================
checkpoint                            release*    revision (pinned)  safe cutoffs (cpo_cutoffs.yaml)
====================================  ==========  ==================  ==========================
google/timesfm-2.5-200m-pytorch       2025-09-02  1d952420fb...       2025-11-28 only (1 of 7)
====================================  ==========  ==================  ==========================
\\* HF ``createdAt``, the earliest publicly-checkable date -- see
``cpo.chronos_baseline`` for why this (not the announcement date) is the
right conservative choice, and why ``revision`` is pinned rather than
resolving ``main`` at call time.

The quantile-column quirk
-----------------------------
``model.forecast()`` returns a ``(batch, horizon, 10)`` array, but the
model's own quantile grid (from the package's ``TimesFM_2p5ModelConfig``) is
only 9 levels, ``[0.1, ..., 0.9]``. Empirically verified (2026-08-14, exact
float equality across all 13 horizons on a real forecast): the point forecast
returned separately by ``forecast()`` is spliced into the quantile array at
column index 5, between q0.5 (index 4) and q0.6 (index 6) -- i.e. the column
order is ``[q0.1, q0.2, q0.3, q0.4, q0.5, MEAN, q0.6, q0.7, q0.8, q0.9]``.
Reading column 5 as q0.6 (an easy off-by-one to make) would silently corrupt
the upper half of every quantile forecast. ``_TIMESFM_QUANTILE_COLUMNS`` below
encodes the verified order once so the mistake can't recur.

The model has no q0.05/q0.95 -- :data:`aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`
needs both. Each is extrapolated linearly from the nearest two native
deciles (0.05 from the q0.1/q0.2 slope, 0.95 from the q0.8/q0.9 slope) rather
than assumed Gaussian, keeping the bulk of the distribution exactly as the
model produced it and only the tails approximate.

Quantile crossing
------------------
``ForecastConfig`` has a ``fix_quantile_crossing`` flag, set ``True`` below --
but read the package source (``timesfm_2p5_torch.py``) before trusting it: its
fix loop only clamps quantile indices 1-4 and 6-9 against the mean at index 5,
and never touches index 0 (q0.1). Verified empirically (2026-08-14, real MPOB
data, with the flag on): q0.1 can still land *above* q0.2, by over 100
currency units on one real horizon -- not a rounding-scale artifact, and not
fixed by the flag despite its name. Indices 1-9 were confirmed monotonic on
that same data (the library's partial fix does work for those), so only index
0 needs a correction: it is clamped to ``min(q0.1, q0.2)`` below, and nothing
else is touched. An earlier version of this fix used
``np.maximum.accumulate`` over the whole 9-level array, which is wrong for
the same reason a first-aid splint on a broken finger should not cover the
whole arm: because q0.1 started out too *high*, accumulating the max forward
dragged q0.2 through q0.5 up to match it, flattening four genuinely distinct,
already-correct quantile levels into a false plateau. Chronos-Bolt was
checked the same way on the same data and has no such gap (see
``cpo.chronos_baseline``), so this is a TimesFM-specific library limitation,
not something to assume of every quantile-head model.

Clamping q0.1 down to ``min(q0.1, q0.2)`` means q0.1 and q0.2 can come out
numerically equal on a horizon where the raw q0.1 was crossed, which in turn
zeroes the slope used to extrapolate q0.05 (so q0.05 == q0.1 == q0.2 too).
That is the honest outcome, not a new bug: once the raw q0.1 is known to be
untrustworthy, asserting anything more specific than "no higher than q0.2"
would be fabricating precision the model never gave us.

Segfault alongside LightGBM
----------------------------
Same underlying issue as :mod:`cpo.chronos_baseline` -- multi-threaded torch
crashes the shared notebook kernel once ``darts_lightgbm`` has already run in
it. ``torch.set_num_threads(1)`` below is the fix; see that module's
docstring for how this was isolated (SIGSEGV, no traceback) and why the
``KMP_DUPLICATE_LIB_OK`` environment variable alone does not fix it.

**Prerequisite:** ``timesfm[torch]`` (a project dependency).
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
_MAX_CONTEXT = 1024
_MAX_HORIZON = 64

# Verified column order of `model.forecast()`'s quantile output -- see module
# docstring. `None` marks the column that is actually the point forecast.
_TIMESFM_QUANTILE_COLUMNS: list[float | None] = [0.1, 0.2, 0.3, 0.4, 0.5, None, 0.6, 0.7, 0.8, 0.9]


def _extrapolate_tail(levels: list[float], values: list[float], target: float) -> float:
    """Linearly extrapolate ``target`` from the two nearest native quantiles."""
    if target < levels[0]:
        x0, x1, y0, y1 = levels[0], levels[1], values[0], values[1]
    else:
        x0, x1, y0, y1 = levels[-2], levels[-1], values[-2], values[-1]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (target - x0)


class WeeklyTimesFMPredictor(Predictor):
    """Zero-shot TimesFM forecast, gated against pretraining leakage.

    Parameters
    ----------
    checkpoint : str
        Hugging Face repo id. Only ``"google/timesfm-2.5-200m-pytorch"`` is
        loadable with the installed package version -- see the module
        docstring.
    revision : str
        Pinned commit sha -- see :mod:`cpo.chronos_baseline` for why.
    release_date : str
        The checkpoint's earliest publicly-checkable date (HF ``createdAt``).
        ``predict()`` returns ``[]`` for any cutoff strictly before this
        date.
    """

    def __init__(self, checkpoint: str, revision: str, release_date: str) -> None:
        self._checkpoint = checkpoint
        self._revision = revision
        self._release_date = pd.Timestamp(release_date)
        self._model: Any = None

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier derived from the checkpoint name."""
        name = self._checkpoint.split("/")[-1].replace("-", "_").replace(".", "_")
        return name if name.startswith("timesfm_") else f"timesfm_{name}"

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

        import timesfm  # noqa: PLC0415
        import torch  # noqa: PLC0415

        torch.set_num_threads(1)  # see cpo.chronos_baseline docstring: avoids a segfault alongside LightGBM

        if self._model is None:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self._checkpoint, revision=self._revision)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=_MAX_CONTEXT,
                    max_horizon=_MAX_HORIZON,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                )
            )
            self._model = model

        values = series.to_numpy(dtype="float32")
        _point, quantile_grid = self._model.forecast(horizon=task.horizon, inputs=[values])
        quantile_grid = quantile_grid[0]  # (horizon, 10)

        native_levels = [lvl for lvl in _TIMESFM_QUANTILE_COLUMNS if lvl is not None]
        native_cols = [i for i, lvl in enumerate(_TIMESFM_QUANTILE_COLUMNS) if lvl is not None]

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []
        for h in task.horizons:
            step = h - 1
            native_values = [float(quantile_grid[step][c]) for c in native_cols]
            native_values[0] = min(native_values[0], native_values[1])  # library gap; see module docstring
            row_quantiles = dict(zip(native_levels, native_values, strict=True))
            for q in STANDARD_QUANTILES:
                if q not in row_quantiles:
                    row_quantiles[q] = _extrapolate_tail(native_levels, native_values, q)
            mean_col = _TIMESFM_QUANTILE_COLUMNS.index(None)
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=float(quantile_grid[step][mean_col]),
                        quantiles={q: float(row_quantiles[q]) for q in STANDARD_QUANTILES},
                    ),
                )
            )
        return predictions


__all__ = ["WeeklyTimesFMPredictor"]

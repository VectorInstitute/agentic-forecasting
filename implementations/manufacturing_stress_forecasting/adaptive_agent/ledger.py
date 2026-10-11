"""Cutoff-aware record of the adaptive agent's past forecasts and their outcomes.

The ledger is what turns strategy mutation into learning from evidence. At each
walk-forward origin it exposes only forecasts whose stress label had been
*published* by that origin, so the agent can confirm or refute a hypothesis
against real outcomes without seeing anything from its own future. Origins are
identified by stable, date-free ids (``origin-001``) so anonymised prompts and
the strategy state never carry calendar dates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class LedgerEntry:
    """One walk-forward forecast and its eventual outcome."""

    origin_id: str
    as_of: pd.Timestamp
    forecast_date: pd.Timestamp
    anchor_probability: float
    probability: float
    outcome: float | None
    outcome_released_at: pd.Timestamp | None
    status: str = "agent"
    signals: dict[str, float] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        """Return a JSON-friendly row for persistence."""
        record = asdict(self)
        for key in ("as_of", "forecast_date", "outcome_released_at"):
            record[key] = None if record[key] is None else pd.Timestamp(record[key]).isoformat()
        return record

    @classmethod
    def from_record(cls, record: dict[str, object]) -> LedgerEntry:
        """Rebuild an entry persisted with ``as_record``."""
        values = dict(record)
        for key in ("as_of", "forecast_date", "outcome_released_at"):
            values[key] = None if values[key] is None else pd.Timestamp(str(values[key]))
        return cls(**values)  # type: ignore[arg-type]


class OutcomeLedger:
    """Track forecasts and reveal outcomes only once they were published."""

    def __init__(self, labels: pd.DataFrame) -> None:
        """Index the full stress-label history by forecast date.

        ``labels`` is the canonical target frame (``timestamp``, ``value``,
        ``released_at``) retrieved with a late cutoff. The ledger never shows a
        label before its ``released_at`` relative to the current origin.
        """
        frame = labels.dropna(subset=["value"])
        self._labels = {
            pd.Timestamp(timestamp): (float(value), pd.Timestamp(released_at))
            for timestamp, value, released_at in zip(
                frame["timestamp"], frame["value"], frame["released_at"], strict=True
            )
        }
        self.entries: list[LedgerEntry] = []
        self.current_as_of: pd.Timestamp | None = None
        self.current_origin_id: str | None = None

    def advance(self, as_of: pd.Timestamp, origin_id: str) -> None:
        """Move the information cutoff to a new origin."""
        if self.current_as_of is not None and as_of <= self.current_as_of:
            raise ValueError(f"Origins must increase; got {as_of} after {self.current_as_of}.")
        self.current_as_of = pd.Timestamp(as_of)
        self.current_origin_id = origin_id

    def record(
        self,
        *,
        origin_id: str,
        as_of: pd.Timestamp,
        forecast_date: pd.Timestamp,
        anchor_probability: float,
        probability: float,
        status: str = "agent",
        signals: dict[str, float] | None = None,
    ) -> LedgerEntry:
        """Store a forecast; its outcome stays hidden until published."""
        outcome, released_at = self._labels.get(pd.Timestamp(forecast_date), (None, None))
        entry = LedgerEntry(
            origin_id=origin_id,
            as_of=pd.Timestamp(as_of),
            forecast_date=pd.Timestamp(forecast_date),
            anchor_probability=float(anchor_probability),
            probability=float(probability),
            outcome=outcome,
            outcome_released_at=released_at,
            status=status,
            signals=dict(signals or {}),
        )
        self.entries.append(entry)
        return entry

    def resolved(self) -> list[LedgerEntry]:
        """Return entries whose outcome was published by the current origin."""
        if self.current_as_of is None:
            return []
        return [
            entry
            for entry in self.entries
            if entry.outcome is not None
            and entry.outcome_released_at is not None
            and entry.outcome_released_at <= self.current_as_of
        ]

    def resolved_origin_ids(self) -> set[str]:
        """Return ids the agent may cite as evidence at the current origin."""
        return {entry.origin_id for entry in self.resolved()}

    def feedback_payload(self, *, recent: int = 12, include_signals: bool = False) -> dict[str, object]:
        """Summarise resolved outcomes for the prompt without calendar dates.

        ``include_signals`` adds the signals seen at each past origin, which a
        review needs to tie misses to conditions it can state as hypotheses.
        """
        resolved = self.resolved()
        rows = [
            {
                "origin_id": entry.origin_id,
                **({"signals_at_origin": entry.signals} if include_signals else {}),
                "your_probability": round(entry.probability, 4),
                "anchor_probability": round(entry.anchor_probability, 4),
                "outcome": int(entry.outcome),  # type: ignore[arg-type]
                "your_brier": round((entry.probability - entry.outcome) ** 2, 4),  # type: ignore[operator]
                "anchor_brier": round((entry.anchor_probability - entry.outcome) ** 2, 4),  # type: ignore[operator]
            }
            for entry in resolved[-recent:]
        ]
        summary: dict[str, object] = {"n_resolved": len(resolved)}
        if resolved:
            summary.update(
                {
                    "n_stress_events": int(sum(entry.outcome for entry in resolved)),  # type: ignore[misc]
                    "your_mean_brier": round(
                        sum((e.probability - e.outcome) ** 2 for e in resolved) / len(resolved),  # type: ignore[operator]
                        5,
                    ),
                    "anchor_mean_brier": round(
                        sum((e.anchor_probability - e.outcome) ** 2 for e in resolved) / len(resolved),  # type: ignore[operator]
                        5,
                    ),
                }
            )
        return {"current_origin_id": self.current_origin_id, "summary": summary, "recent_resolved": rows}

    def resolved_since(self, origin_ids: set[str]) -> list[LedgerEntry]:
        """Return resolved entries not among ``origin_ids`` (e.g. already reviewed)."""
        return [entry for entry in self.resolved() if entry.origin_id not in origin_ids]


__all__ = ["LedgerEntry", "OutcomeLedger"]

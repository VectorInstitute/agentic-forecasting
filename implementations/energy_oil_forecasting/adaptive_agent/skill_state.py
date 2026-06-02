"""WTI forecasting strategy state model.

Defines the structured state backing the ``wti-strategy`` adaptive skill.
``WtiStrategyState`` is the single source of truth for the agent's current
forecasting approach.  It is persisted to ``skills/wti-strategy/skill_state.yaml``
and rendered to ``skills/wti-strategy/SKILL.md`` on every mutation so that the
ADK ``SkillToolset`` always reads an up-to-date version.

Learning layers
---------------
The four fields of ``WtiStrategyState`` map to distinct update frequencies and
evidence burdens, enforced partly by the mutation tools in ``skill_tools.py``
and partly by the ``meta-learning`` governance skill:

``observations``
    Append-only log of pattern-level findings.  Lowest evidence bar — record
    any finding that is not a single-outlier surprise.

``hypotheses``
    Candidate systematic corrections the agent is actively testing.  Open a
    hypothesis when you suspect a durable pattern.  Accumulate confirmation /
    refutation counts across resolutions.  A hypothesis graduates to a
    calibration correction when its confirmation count reaches the store's
    ``confirmation_threshold``.

``calibration_corrections``
    Confirmed systematic adjustments applied at prediction time.  Each entry
    is graduated from a confirmed hypothesis — never added directly.

``approach_narrative``
    Free-text description of the agent's overall forecasting philosophy.
    Highest evidence bar.  Update only when the calibration record reveals a
    structural insight that the narrative no longer captures.
"""

from __future__ import annotations

from typing import Literal

from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillState
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """A single pattern-level finding from a resolution or self-review."""

    date: str
    finding: str
    linked_hypothesis: str | None = None


class Hypothesis(BaseModel):
    """A candidate systematic correction under active testing.

    ``status`` progresses through ``open`` → ``confirmed`` or ``open`` →
    ``refuted``.  Confirmed hypotheses are graduated to
    :class:`CalibrationCorrection` via the ``graduate_hypothesis`` tool.
    """

    id: str
    claim: str
    status: Literal["open", "confirmed", "refuted"] = "open"
    confirmations: int = 0
    refutations: int = 0
    opened_on: str


class CalibrationCorrection(BaseModel):
    """A confirmed systematic adjustment applied at prediction time.

    Every entry here was graduated from a confirmed hypothesis; the
    ``source_hypothesis`` field preserves that lineage.
    """

    condition: str
    adjustment: str
    horizon_scope: str
    source_hypothesis: str
    confirmed_on: str


class VersionEntry(BaseModel):
    """One row in the version history table."""

    date: str
    description: str


# ---------------------------------------------------------------------------
# Strategy state
# ---------------------------------------------------------------------------


class WtiStrategyState(AdaptiveSkillState):
    """Structured state for the adaptive WTI crude oil forecasting strategy.

    See module docstring for the learning-layer hierarchy and evidence burdens.
    """

    approach_narrative: str
    calibration_corrections: list[CalibrationCorrection] = []
    hypotheses: list[Hypothesis] = []
    observations: list[Observation] = []
    version_history: list[VersionEntry] = []

    def build_markdown(self, skill_name: str | None = None) -> str:  # noqa: PLR0912
        """Render the full ``SKILL.md`` content from current state."""
        lines: list[str] = []

        # Frontmatter — skill_name must match the containing directory name (ADK requirement)
        lines += [
            "---",
            f"name: {skill_name or 'wti-strategy'}",
            "description: >-",
            "  The adaptive WTI analyst's current forecasting strategy. Load this at the",
            "  start of every prediction task. This file is generated — edit the state",
            "  through the mutation tools, not by hand.",
            "---",
            "",
        ]

        lines += [
            "# WTI Forecasting Strategy",
            "",
            "## Approach",
            "",
            self.approach_narrative.strip(),
            "",
        ]

        # Active calibration corrections
        lines += [
            "## Active calibration corrections",
            "",
        ]
        if self.calibration_corrections:
            lines += [
                "| Condition | Adjustment | Horizon scope | Confirmed on |",
                "|-----------|-----------|---------------|--------------|",
            ]
            for c in self.calibration_corrections:
                lines.append(f"| {c.condition} | {c.adjustment} | {c.horizon_scope} | {c.confirmed_on} |")
        else:
            lines.append("*(No calibration corrections yet. Graduate a confirmed hypothesis to add one.)*")
        lines.append("")

        # Open hypotheses
        lines += [
            "## Open hypotheses",
            "",
        ]
        open_hyps = [h for h in self.hypotheses if h.status == "open"]
        if open_hyps:
            lines += [
                "| ID | Claim | Confirmations | Refutations |",
                "|----|-------|---------------|-------------|",
            ]
            for h in open_hyps:
                lines.append(f"| {h.id} | {h.claim} | {h.confirmations} | {h.refutations} |")
        else:
            lines.append("*(No open hypotheses.)*")
        lines.append("")

        # Closed hypotheses (confirmed / refuted) — collapsed for readability
        closed_hyps = [h for h in self.hypotheses if h.status != "open"]
        if closed_hyps:
            lines += [
                "## Closed hypotheses",
                "",
                "| ID | Claim | Status | Confirmations | Refutations |",
                "|----|-------|--------|---------------|-------------|",
            ]
            for h in closed_hyps:
                lines.append(f"| {h.id} | {h.claim} | {h.status} | {h.confirmations} | {h.refutations} |")
            lines.append("")

        # Observations
        lines += [
            "## Observations",
            "",
        ]
        if self.observations:
            lines += [
                "| Date | Finding | Linked hypothesis |",
                "|------|---------|-------------------|",
            ]
            for o in self.observations:
                linked = o.linked_hypothesis or "—"
                lines.append(f"| {o.date} | {o.finding} | {linked} |")
        else:
            lines.append("*(No observations yet. Record findings from resolutions and self-reviews.)*")
        lines.append("")

        # Version history
        lines += [
            "## Version history",
            "",
            "| Date | Change |",
            "|------|--------|",
        ]
        for v in self.version_history:
            lines.append(f"| {v.date} | {v.description} |")
        lines.append("")

        return "\n".join(lines)

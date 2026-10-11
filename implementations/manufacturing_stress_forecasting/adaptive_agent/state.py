from __future__ import annotations

from typing import Literal

from aieng.forecasting.methods.agentic import AdaptiveSkillState
from pydantic import BaseModel, Field


class ManufacturingObservation(BaseModel):
    date: str
    finding: str
    linked_hypothesis: str | None = None


class ManufacturingHypothesis(BaseModel):
    id: str
    claim: str
    status: Literal["open", "confirmed", "refuted"] = "open"
    confirmations: int = 0
    refutations: int = 0
    origins: list[str] = Field(default_factory=list)


class ManufacturingCalibrationCorrection(BaseModel):
    condition: str
    adjustment: str
    source_hypothesis: str
    confirmed_on: str


class ManufacturingStrategyState(AdaptiveSkillState):
    approach_narrative: str = "Start from the numerical hybrid anchor and make only bounded, evidence-backed adjustments."
    calibration_corrections: list[ManufacturingCalibrationCorrection] = Field(default_factory=list)
    hypotheses: list[ManufacturingHypothesis] = Field(default_factory=list)
    observations: list[ManufacturingObservation] = Field(default_factory=list)

    def build_markdown(self, skill_name: str | None = None) -> str:
        lines = [
            "---",
            f"name: {skill_name or 'manufacturing-strategy'}",
            "description: >-",
            "  Governed adaptive manufacturing-stress forecasting strategy.",
            "---",
            "",
            "# Manufacturing Stress Strategy",
            "",
            "## Approach",
            "",
            self.approach_narrative,
            "",
            "## Calibration corrections",
            "",
        ]
        if self.calibration_corrections:
            lines += ["| Condition | Adjustment | Source |", "|---|---|---|"]
            lines.extend(f"| {c.condition} | {c.adjustment} | {c.source_hypothesis} |" for c in self.calibration_corrections)
        else:
            lines.append("*(None graduated.)*")
        lines += ["", "## Open hypotheses", ""]
        open_hypotheses = [h for h in self.hypotheses if h.status == "open"]
        lines.extend(
            [f"- `{h.id}`: {h.claim} ({h.confirmations} confirmations, {h.refutations} refutations)" for h in open_hypotheses]
            or ["*(None.)*"]
        )
        lines += ["", "## Observations", ""]
        lines.extend([f"- {o.date}: {o.finding}" for o in self.observations] or ["*(None.)*"])
        return "\n".join(lines) + "\n"

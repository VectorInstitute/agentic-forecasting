from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from aieng.forecasting.methods.agentic import AdaptiveSkillStore
from manufacturing_stress_forecasting.adaptive_agent.ledger import OutcomeLedger
from manufacturing_stress_forecasting.adaptive_agent.state import (
    ManufacturingCalibrationCorrection,
    ManufacturingHypothesis,
    ManufacturingObservation,
    ManufacturingStrategyState,
)


def _state_hash(state: ManufacturingStrategyState) -> str:
    payload = json.dumps(state.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audit_mutation(
    audit_path: Path,
    *,
    tool: str,
    strategy_dir: Path,
    state_before: ManufacturingStrategyState,
    state_after: ManufacturingStrategyState,
    result: str,
    origin_id: str | None = None,
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "strategy_mutation",
        "tool": tool,
        "strategy_path": strategy_dir.name,
        "strategy_version": _state_hash(state_after)[:12],
        "result": result,
        "state_before_sha256": _state_hash(state_before),
        "state_after_sha256": _state_hash(state_after),
    }
    if origin_id is not None:
        event["origin_id"] = origin_id
    with audit_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, separators=(",", ":")) + "\n")


def build_strategy_tools(  # noqa: PLR0915 - one closure per typed tool
    strategy_dir: Path,
    *,
    confirmation_threshold: int = 3,
    audit_path: Path | None = None,
    ledger: OutcomeLedger | None = None,
) -> list[Callable[..., str]]:
    """Build the typed strategy-mutation tools.

    Hypothesis outcomes are only accepted against forecast origins whose stress
    outcome the ``ledger`` shows as published at the current origin. Without a
    ledger there is no resolved evidence, so outcomes are refused; this blocks
    the agent from "confirming" its own claims within a single forecast.
    """
    store = AdaptiveSkillStore(strategy_dir, ManufacturingStrategyState, confirmation_threshold=confirmation_threshold)
    resolved_audit_path = audit_path or strategy_dir / ".history" / "adaptation_audit.jsonl"

    def stamp() -> str:
        if ledger is not None and ledger.current_origin_id is not None:
            return ledger.current_origin_id
        return str(date.today())

    def save_mutation(
        tool: str,
        before: ManufacturingStrategyState,
        after: ManufacturingStrategyState,
        result: str,
    ) -> str:
        store.save(after)
        _audit_mutation(
            resolved_audit_path,
            tool=tool,
            strategy_dir=strategy_dir,
            state_before=before,
            state_after=after,
            result=result,
            origin_id=ledger.current_origin_id if ledger is not None else None,
        )
        return result

    def record_observation(finding: str, linked_hypothesis: str = "") -> str:
        """Record a durable observation about how forecasts have been performing.

        Args:
            finding: One concise, evidence-backed sentence without calendar dates.
            linked_hypothesis: Optional hypothesis id (for example ``hyp-001``) this observation supports.
        """
        state = store.load()
        before = state.model_copy(deep=True)
        state.observations.append(
            ManufacturingObservation(
                date=stamp(), finding=finding.strip(), linked_hypothesis=linked_hypothesis.strip() or None
            )
        )
        return save_mutation("record_observation", before, state, "Manufacturing observation recorded.")

    def open_hypothesis(claim: str, initial_evidence: str) -> str:
        """Open a testable hypothesis about when the numerical anchor is miscalibrated.

        Args:
            claim: A condition-and-direction claim, e.g. "When X holds, the anchor under-forecasts stress."
            initial_evidence: The resolved outcomes that motivated the claim, citing their origin ids.
        """
        state = store.load()
        before = state.model_copy(deep=True)
        hypothesis_id = f"hyp-{len(state.hypotheses) + 1:03d}"
        state.hypotheses.append(ManufacturingHypothesis(id=hypothesis_id, claim=claim.strip()))
        state.observations.append(
            ManufacturingObservation(date=stamp(), finding=initial_evidence.strip(), linked_hypothesis=hypothesis_id)
        )
        return save_mutation(
            "open_hypothesis",
            before,
            state,
            f"Opened {hypothesis_id}; {store.confirmation_threshold} confirmations from distinct resolved "
            "origins are required before graduation.",
        )

    def record_hypothesis_outcome(hypothesis_id: str, outcome: str, origin: str) -> str:
        """Score an open hypothesis against one resolved forecast origin.

        Args:
            hypothesis_id: The open hypothesis id, e.g. ``hyp-001``.
            outcome: ``confirmed`` or ``refuted``.
            origin: An ``origin_id`` listed in ``resolved_feedback.recent_resolved``.
        """
        if outcome not in {"confirmed", "refuted"}:
            return "Outcome must be 'confirmed' or 'refuted'."
        if ledger is None:
            return (
                "Refused: no resolved forecast outcomes are available in this run, so hypotheses cannot be "
                "confirmed or refuted. Record an observation instead."
            )
        allowed = ledger.resolved_origin_ids()
        if origin not in allowed:
            examples = ", ".join(sorted(allowed)[-5:]) or "none yet"
            return f"Refused: {origin!r} is not a resolved origin. Resolved origin ids include: {examples}."
        state = store.load()
        before = state.model_copy(deep=True)
        hypothesis = next((item for item in state.hypotheses if item.id == hypothesis_id), None)
        if hypothesis is None or hypothesis.status != "open":
            return f"Open hypothesis {hypothesis_id!r} was not found."
        if origin in hypothesis.origins:
            return f"Origin {origin!r} has already been recorded for {hypothesis_id}."
        hypothesis.origins.append(origin)
        if outcome == "confirmed":
            hypothesis.confirmations += 1
        else:
            hypothesis.refutations += 1
        message = f"Recorded {outcome} outcome for {hypothesis_id}."
        if hypothesis.refutations >= store.confirmation_threshold and hypothesis.refutations > hypothesis.confirmations:
            hypothesis.status = "refuted"
            message += f" {hypothesis_id} is now refuted and closed."
        return save_mutation("record_hypothesis_outcome", before, state, message)

    def graduate_hypothesis(hypothesis_id: str, condition: str, adjustment: str) -> str:
        """Promote a well-confirmed hypothesis into a standing calibration correction.

        Args:
            hypothesis_id: A hypothesis with enough confirmations from distinct resolved origins.
            condition: When the correction applies, stated in terms of supplied signals.
            adjustment: The bounded direction and size of the adjustment to the anchor.
        """
        state = store.load()
        before = state.model_copy(deep=True)
        hypothesis = next((item for item in state.hypotheses if item.id == hypothesis_id), None)
        if hypothesis is None:
            return f"Hypothesis {hypothesis_id!r} was not found."
        if hypothesis.status != "open":
            return f"Cannot graduate {hypothesis_id}: its status is {hypothesis.status!r}."
        if hypothesis.confirmations < store.confirmation_threshold:
            return f"Cannot graduate {hypothesis_id}: {hypothesis.confirmations}/{store.confirmation_threshold} confirmations."
        if len(set(hypothesis.origins)) < store.confirmation_threshold:
            return f"Cannot graduate {hypothesis_id}: confirmations must come from distinct origins."
        if hypothesis.refutations >= hypothesis.confirmations:
            return f"Cannot graduate {hypothesis_id}: refutations ({hypothesis.refutations}) match or exceed confirmations."
        hypothesis.status = "confirmed"
        state.calibration_corrections.append(
            ManufacturingCalibrationCorrection(
                condition=condition.strip(),
                adjustment=adjustment.strip(),
                source_hypothesis=hypothesis_id,
                confirmed_on=stamp(),
            )
        )
        return save_mutation(
            "graduate_hypothesis", before, state, f"Graduated {hypothesis_id} into a calibration correction."
        )

    def update_approach_narrative(new_text: str, rationale: str) -> str:
        """Rewrite the overall forecasting approach when resolved evidence justifies it.

        Args:
            new_text: The new approach, in at most three sentences and without calendar dates.
            rationale: The resolved evidence behind the change.
        """
        state = store.load()
        before = state.model_copy(deep=True)
        state.approach_narrative = f"{new_text.strip()} Rationale: {rationale.strip()}"
        return save_mutation("update_approach_narrative", before, state, "Manufacturing approach narrative updated.")

    return [
        record_observation,
        open_hypothesis,
        record_hypothesis_outcome,
        graduate_hypothesis,
        update_approach_narrative,
    ]

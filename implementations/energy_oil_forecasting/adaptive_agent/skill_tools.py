"""Mutation tools for the ``wti-strategy`` adaptive skill.

These are plain Python callables registered as ADK ``FunctionTool`` objects via
``AgentConfig(extra_tools=build_skill_tools(strategy_dir))``.  They run in the
host process — *not* inside the E2B sandbox — so they can read and write the
skill directory on the local filesystem.

Factory pattern
---------------
Use :func:`build_skill_tools` to create a set of tools bound to a specific
strategy directory.  This allows multiple named strategy variants (e.g.
``wti-strategy-stats``, ``wti-strategy-news``) to coexist, each with its own
``AdaptiveSkillStore`` and ``skill_state.yaml``.

The module-level :data:`STORE` and :data:`WTI_SKILL_TOOLS` are convenience
bindings to the **default** ``wti-strategy`` directory for backward
compatibility and interactive ``adk web`` use.

Design principles
-----------------
Each tool follows the same three-step cycle:

1. ``store.load()`` — deserialise current state from ``skill_state.yaml``.
2. Apply one typed mutation to the state model.
3. ``store.save(state)`` — write YAML, re-render ``SKILL.md``, back up.

Tool signatures are intentionally narrow: they accept only the arguments
needed for one specific mutation.  The agent cannot write arbitrary content to
the skill directory through any of these tools.

Evidence governance
-------------------
``record_observation``
    No guard.  Record any pattern-level finding (not a single-outlier surprise).

``open_hypothesis``
    No guard.  Open a hypothesis whenever you suspect a durable pattern.

``record_hypothesis_outcome``
    Validates the hypothesis ID exists and is still open.

``graduate_hypothesis``
    Hard guard: rejected if ``hypothesis.confirmations < store.confirmation_threshold``.
    Returns a clear message stating the shortfall.

``update_approach_narrative``
    Requires a ``rationale`` argument but no hard numeric guard.  The
    ``meta-learning`` skill governs when this is appropriate.

Scope guard
-----------
All writes go through the :class:`~aieng.forecasting.methods.agentic.adaptive_skill.AdaptiveSkillStore`
instance passed to :func:`build_skill_tools`.  No path outside that directory
can be reached.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore
from energy_oil_forecasting.adaptive_agent.skill_state import (
    CalibrationCorrection,
    Hypothesis,
    Observation,
    VersionEntry,
    WtiStrategyState,
)


# ---------------------------------------------------------------------------
# Default strategy directory (for backward compat and adk web)
# ---------------------------------------------------------------------------

_SKILL_DIR = Path(__file__).parent / "skills" / "wti-strategy"


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------


def _today() -> str:
    return str(date.today())


def _next_hypothesis_id(state: WtiStrategyState) -> str:
    """Return the next sequential hypothesis ID (e.g. ``hyp-004``)."""
    n = len(state.hypotheses) + 1
    return f"hyp-{n:03d}"


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def build_skill_tools(  # noqa: PLR0915
    strategy_dir: Path,
    *,
    confirmation_threshold: int = 3,
) -> list[Callable[..., str]]:
    """Build a set of strategy mutation tools bound to *strategy_dir*.

    Each call returns five fresh callables (closures over a new
    :class:`~aieng.forecasting.methods.agentic.adaptive_skill.AdaptiveSkillStore`
    instance).  Pass the returned list to
    ``AgentConfig(extra_tools=build_skill_tools(strategy_dir))`` to wire the
    tools into an agent that operates on a specific strategy variant.

    Parameters
    ----------
    strategy_dir : Path
        Directory containing the strategy skill (``skill_state.yaml``,
        ``SKILL.md``, ``.history/``).  Must exist and be a directory.
    confirmation_threshold : int, default=3
        Number of confirming hypothesis outcomes required before
        ``graduate_hypothesis`` is permitted.

    Returns
    -------
    list[Callable[..., str]]
        ``[record_observation, open_hypothesis, record_hypothesis_outcome,
        graduate_hypothesis, update_approach_narrative]``
    """
    store: AdaptiveSkillStore[WtiStrategyState] = AdaptiveSkillStore(
        skill_dir=strategy_dir,
        state_type=WtiStrategyState,
        confirmation_threshold=confirmation_threshold,
    )

    def record_observation(finding: str, linked_hypothesis: str = "") -> str:
        """Record a pattern-level finding from a resolution or self-review.

        Call this whenever you observe a systematic pattern across multiple
        forecasts — not after a single surprising outcome.

        Parameters
        ----------
        finding : str
            A concise description of the pattern observed.  Be specific: include
            the regime, horizon, and direction of the error where applicable.
            Example: "80% intervals missed 4 of 5 actuals in the elevated vol
            regime at the 21-day horizon."
        linked_hypothesis : str, optional
            ID of an existing open hypothesis this observation supports or
            refutes (e.g. ``"hyp-001"``).  Leave blank if this is a fresh
            observation not yet linked to any hypothesis.

        Returns
        -------
        str
            Confirmation message.
        """
        state = store.load()
        obs = Observation(
            date=_today(),
            finding=finding.strip(),
            linked_hypothesis=linked_hypothesis.strip() or None,
        )
        state.observations.append(obs)
        store.save(state)
        linked_note = f" (linked to {obs.linked_hypothesis})" if obs.linked_hypothesis else ""
        return f'Observation recorded{linked_note}: "{finding[:80]}{"..." if len(finding) > 80 else ""}"'

    def open_hypothesis(claim: str, initial_evidence: str) -> str:
        """Open a new hypothesis about a suspected systematic forecasting pattern.

        A hypothesis is a candidate calibration correction under active testing.
        Open one when you have at least one observation suggesting a durable
        pattern but do not yet have enough confirming resolutions to graduate it.

        Parameters
        ----------
        claim : str
            A testable claim about your forecasting behaviour.  State it in terms
            of a specific condition and a directional error.
            Example: "My 80% prediction intervals are consistently too narrow
            when the vol regime is classified as elevated or extreme."
        initial_evidence : str
            The observation(s) that motivated opening this hypothesis.  This is
            for the audit record — be specific about the number of data points.

        Returns
        -------
        str
            Confirmation message including the assigned hypothesis ID.
        """
        state = store.load()
        hyp_id = _next_hypothesis_id(state)
        hyp = Hypothesis(
            id=hyp_id,
            claim=claim.strip(),
            status="open",
            confirmations=0,
            refutations=0,
            opened_on=_today(),
        )
        state.hypotheses.append(hyp)
        obs = Observation(
            date=_today(),
            finding=initial_evidence.strip(),
            linked_hypothesis=hyp_id,
        )
        state.observations.append(obs)
        store.save(state)
        return (
            f'Hypothesis {hyp_id} opened: "{claim[:80]}{"..." if len(claim) > 80 else ""}". '
            f"Initial evidence recorded as an observation linked to {hyp_id}. "
            f"Confirmations needed to graduate: {store.confirmation_threshold}."
        )

    def record_hypothesis_outcome(hypothesis_id: str, outcome: str) -> str:
        """Record a confirming or refuting outcome for an open hypothesis.

        Call this after each resolution where the outcome is directly relevant
        to an open hypothesis.  Accumulate enough confirmations to graduate.

        Parameters
        ----------
        hypothesis_id : str
            ID of the hypothesis to update (e.g. ``"hyp-001"``).
        outcome : str
            Either ``"confirmed"`` or ``"refuted"``.  A single refutation does
            not automatically close the hypothesis — continue accumulating
            evidence.  A hypothesis should be manually closed (status →
            ``"refuted"``) only when refutations clearly outweigh confirmations
            across a meaningful sample.

        Returns
        -------
        str
            Updated confirmation / refutation counts and progress toward the
            graduation threshold.
        """
        if outcome not in ("confirmed", "refuted"):
            return f"Invalid outcome '{outcome}'. Must be 'confirmed' or 'refuted'."

        state = store.load()
        hyp = next((h for h in state.hypotheses if h.id == hypothesis_id), None)
        if hyp is None:
            ids = [h.id for h in state.hypotheses]
            return f"Hypothesis '{hypothesis_id}' not found. Known IDs: {ids}."
        if hyp.status != "open":
            return f"Hypothesis {hypothesis_id} is already {hyp.status}. Only open hypotheses can receive new outcomes."

        if outcome == "confirmed":
            hyp.confirmations += 1
        else:
            hyp.refutations += 1

        store.save(state)

        remaining = max(0, store.confirmation_threshold - hyp.confirmations)
        if remaining == 0:
            ready_msg = (
                f" Ready to graduate — call graduate_hypothesis('{hypothesis_id}', ...) "
                "with a condition, adjustment, and horizon_scope."
            )
        else:
            ready_msg = f" {remaining} more confirmation(s) needed to graduate."

        return (
            f"{hypothesis_id} updated: {hyp.confirmations} confirmation(s), {hyp.refutations} refutation(s).{ready_msg}"
        )

    def graduate_hypothesis(
        hypothesis_id: str,
        condition: str,
        adjustment: str,
        horizon_scope: str,
    ) -> str:
        """Graduate a confirmed hypothesis to an active calibration correction.

        This is the primary mechanism through which the agent's strategy
        improves.  A calibration correction is applied at every future
        prediction; it is not merely recorded — it changes behaviour.

        This tool enforces the confirmation threshold: it will reject the call
        if the hypothesis has not accumulated enough confirming outcomes.

        Parameters
        ----------
        hypothesis_id : str
            ID of the confirmed hypothesis to graduate (e.g. ``"hyp-001"``).
        condition : str
            The specific condition under which this correction applies.
            Example: "vol regime is elevated or extreme".
        adjustment : str
            The concrete adjustment to make when the condition is met.
            Example: "Widen 80% CI by 12% relative to the statistical model
            output."
        horizon_scope : str
            Which horizons this correction applies to.
            One of: ``"all"``, ``"5bd"``, ``"10bd"``, ``"21bd"``, or a
            combination like ``"10bd and 21bd"``.

        Returns
        -------
        str
            Confirmation message, or a rejection message with the shortfall.
        """
        state = store.load()
        hyp = next((h for h in state.hypotheses if h.id == hypothesis_id), None)
        if hyp is None:
            ids = [h.id for h in state.hypotheses]
            return f"Hypothesis '{hypothesis_id}' not found. Known IDs: {ids}."
        if hyp.status != "open":
            return f"Hypothesis {hypothesis_id} is already {hyp.status}. Only open hypotheses can be graduated."

        if hyp.confirmations < store.confirmation_threshold:
            shortfall = store.confirmation_threshold - hyp.confirmations
            return (
                f"Cannot graduate {hypothesis_id}: "
                f"{hyp.confirmations} confirmation(s), "
                f"requires {store.confirmation_threshold}. "
                f"Record {shortfall} more confirming outcome(s) first."
            )

        today = _today()
        hyp.status = "confirmed"
        correction = CalibrationCorrection(
            condition=condition.strip(),
            adjustment=adjustment.strip(),
            horizon_scope=horizon_scope.strip(),
            source_hypothesis=hypothesis_id,
            confirmed_on=today,
        )
        state.calibration_corrections.append(correction)
        state.observations.append(
            Observation(
                date=today,
                finding=(
                    f"Graduated {hypothesis_id} to calibration correction: "
                    f"'{condition}' → '{adjustment}' ({horizon_scope})."
                ),
                linked_hypothesis=hypothesis_id,
            )
        )
        state.version_history.append(
            VersionEntry(
                date=today,
                description=(
                    f"Graduated {hypothesis_id} to calibration correction "
                    f"(condition: {condition[:50]}{'...' if len(condition) > 50 else ''})."
                ),
            )
        )
        store.save(state)
        return (
            f"Hypothesis {hypothesis_id} confirmed and graduated. "
            f"Calibration correction added: when '{condition}', apply '{adjustment}' "
            f"(scope: {horizon_scope})."
        )

    def update_approach_narrative(new_text: str, rationale: str) -> str:
        """Replace the approach narrative with an updated strategic description.

        This is the highest-evidence-bar update.  Consult ``meta-learning``
        before calling this tool — the narrative should only change when the
        calibration record reveals a structural insight that the current
        description no longer captures.  A ``rationale`` argument is required
        to force articulation of why the change is warranted.

        Parameters
        ----------
        new_text : str
            The complete replacement text for the ``## Approach`` section.
            Write it as a self-contained description of the current forecasting
            strategy — what signals are used, in what order, and with what
            emphasis.
        rationale : str
            Why this update is warranted now.  Cite the specific calibration
            corrections or pattern of observations that motivated the change.

        Returns
        -------
        str
            Confirmation message.
        """
        if not new_text.strip():
            return "new_text must not be empty."
        if not rationale.strip():
            return "rationale must not be empty. Explain why the approach narrative warrants an update."

        state = store.load()
        today = _today()
        state.approach_narrative = new_text.strip()
        state.version_history.append(
            VersionEntry(
                date=today,
                description=f"Updated approach narrative. Rationale: {rationale[:120]}{'...' if len(rationale) > 120 else ''}",
            )
        )
        store.save(state)
        return f"Approach narrative updated ({len(new_text)} chars). Rationale recorded in version history."

    return [
        record_observation,
        open_hypothesis,
        record_hypothesis_outcome,
        graduate_hypothesis,
        update_approach_narrative,
    ]


# ---------------------------------------------------------------------------
# Backward-compatible module-level bindings (default wti-strategy dir)
# ---------------------------------------------------------------------------

STORE: AdaptiveSkillStore[WtiStrategyState] = AdaptiveSkillStore(
    skill_dir=_SKILL_DIR,
    state_type=WtiStrategyState,
    confirmation_threshold=3,
)

WTI_SKILL_TOOLS: list[Callable[..., str]] = build_skill_tools(_SKILL_DIR)

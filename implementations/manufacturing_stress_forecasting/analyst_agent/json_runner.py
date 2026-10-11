"""Manufacturing-local tolerance for structured agent output."""

from __future__ import annotations

import ast
import json
from typing import Any

from aieng.forecasting.methods.agentic import AdkTextRunner
from google.adk.agents.run_config import RunConfig


def normalize_json_object(output: str) -> str:
    """Convert a Python dict literal to JSON while leaving other text untouched.

    Some proxy/ADK responses contain the correct structured fields but serialize
    them with Python's single-quoted dict representation. ast.literal_eval
    safely handles that narrow case without executing arbitrary model output.
    """
    try:
        json.loads(output)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(output)
        except (SyntaxError, ValueError):
            return output

        if not isinstance(parsed, dict):
            return output
        return json.dumps(parsed, separators=(",", ":"))

    return output


class ManufacturingStressJsonRunner(AdkTextRunner):
    """Normalize a narrow proxy formatting edge case before schema validation."""

    async def run_text_async(
        self,
        prompt: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        run_config: RunConfig | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> str:
        """Run one agent turn and return a JSON-compatible object string."""
        output = await super().run_text_async(
            prompt,
            user_id=user_id,
            session_id=session_id,
            run_config=run_config,
            initial_state=initial_state,
        )
        return normalize_json_object(output)


__all__ = ["ManufacturingStressJsonRunner", "normalize_json_object"]

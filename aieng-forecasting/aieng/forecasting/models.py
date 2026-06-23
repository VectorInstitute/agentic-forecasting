"""Canonical proxy model identifiers used across the project.

The bootcamp standardizes on exactly two Vector-proxy models so examples,
defaults, and notebooks stay consistent for participants:

- :data:`LITE_MODEL` — the default / lite model. Fast and cheap; used
  everywhere unless a task specifically benefits from the advanced model.
- :data:`ADVANCED_MODEL` — the advanced model. Higher capability; reserved for
  the adaptive-agent path and production-quality / curriculum-generation runs.

Reference these constants instead of hardcoding model strings, so a model
swap is a one-line change here rather than a repo-wide find-and-replace.

This module is intentionally dependency-free (it imports nothing from the rest
of the package) so it can be imported from anywhere without risking an import
cycle.
"""

from __future__ import annotations


#: Default / lite model — fast and cheap; the project-wide default.
LITE_MODEL = "gemini-3.1-flash-lite-preview"

#: Advanced model — higher capability; adaptive-agent and production runs.
ADVANCED_MODEL = "gemini-3.5-flash"

#: Alias for the project-wide default model (the lite model).
DEFAULT_MODEL = LITE_MODEL


__all__ = ["ADVANCED_MODEL", "DEFAULT_MODEL", "LITE_MODEL"]

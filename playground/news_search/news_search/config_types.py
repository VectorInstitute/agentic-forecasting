"""Pydantic config models loaded from YAML."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for the Google ADK LlmAgent."""

    model: str = "gemini-2.0-flash"
    system_prompt: str = (
        "You are a news research assistant. "
        "Search for and summarize major global news headlines for the specified date."
    )
    temperature: float = 0.1
    max_output_tokens: int = 4096


class DateRangeConfig(BaseModel):
    """Inclusive date range to iterate over."""

    start: date
    end: date


class RunConfig(BaseModel):
    """Top-level configuration loaded from YAML."""

    id: str = Field(description="Stable run identifier used in Langfuse session names.")
    display_label: str
    description: str = ""

    langfuse_dataset_name: str = Field(
        description="Langfuse dataset name. Use a fresh name when task questions change."
    )

    date_range: DateRangeConfig
    max_dates: int | None = Field(
        default=None,
        description="Cap the number of dates processed. Useful for smoke tests. Remove or set null for a full run.",
    )

    agent: AgentConfig = Field(default_factory=AgentConfig)

    task_prompt_template: str = Field(
        default=(
            "Please search for and summarize the major global news headlines from {date_long} ({date_iso}).\n\n"
            "Focus specifically on:\n"
            "- Breaking news and major events published on or before {date_iso}\n"
            "- Political and geopolitical developments\n"
            "- Economic and financial news\n"
            "- Significant cultural or scientific events\n\n"
            "Return a numbered list of the top 5-10 headlines, each with a 1-2 sentence summary "
            "and the source/publication if you can identify it.\n\n"
            "IMPORTANT: Only include events that were publicly known on {date_iso}. "
            "Do not include anything that happened after that date."
        )
    )

    stride_days: int = Field(
        default=1,
        ge=1,
        description=(
            "Step size between sampled dates. "
            "stride_days=1 samples every day; stride_days=7 samples once per week."
        ),
    )

    run_name_prefix: str = "news-grounding"

    output_dir: str | None = Field(
        default="outputs",
        description=(
            "Directory to write per-date markdown summaries into.  Each run gets its own "
            "subdirectory: <output_dir>/<run_name>/<date_iso>.md.  "
            "Set to null to disable filesystem output."
        ),
    )

    delay_between_requests_sec: float = Field(
        default=5.0,
        description=(
            "Seconds to wait between successive agent calls. "
            "Increase if you hit 429 rate-limit errors from the Gemini API."
        ),
    )

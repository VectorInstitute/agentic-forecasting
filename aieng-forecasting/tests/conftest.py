"""Suite-wide test configuration.

Disables Langfuse tracing for the whole test suite. The LLM-call seam emits a
Langfuse generation per completion (see
``aieng.forecasting.methods.llm_processes._client.langfuse_generation``), and
several suites load real credentials from the repo-root ``.env``. Without this
guard, unit tests with a mocked ``litellm`` would ship real, zero-usage
observations to the live Langfuse project.

Set at import time rather than in a fixture so it lands before any test module
constructs a Langfuse client.
"""

from __future__ import annotations

import os


os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

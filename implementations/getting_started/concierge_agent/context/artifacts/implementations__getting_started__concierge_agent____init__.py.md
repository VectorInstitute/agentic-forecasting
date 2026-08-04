# Source: implementations/getting_started/concierge_agent/__init__.py

kind: python

```python
"""Repo concierge agent — onboarding helper for the agentic-forecasting codebase.

Exports the :class:`AgentConfig` factory and the knowledge-search tool. Pair
with ``99_repo_concierge.ipynb`` or ``adk run implementations/getting_started/concierge_agent``
from the repository root.
"""

from getting_started.concierge_agent.agent import build_concierge_config
from getting_started.concierge_agent.catalog import (
    fetch_repo_artifact,
    search_repo_catalog,
)
from getting_started.concierge_agent.knowledge import search_repo_knowledge


__all__ = [
    "build_concierge_config",
    "fetch_repo_artifact",
    "search_repo_catalog",
    "search_repo_knowledge",
]
```

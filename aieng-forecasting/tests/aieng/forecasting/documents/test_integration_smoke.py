"""Integration smoke test: DocumentStore → ForecastContext → LLMP prompt.

Verifies the full pipeline from real CFPR extracted report artifacts through
to the prompt building step, without hitting a live LLM API.  Requires that
``scripts/extract_reports.py`` has been run to populate ``data/reports/cfpr/``.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.store import SeriesStore
from aieng.forecasting.documents.store import DocumentStore
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.llm_processes.base import (
    LLMPredictorConfig,
    build_report_preamble,
    fetch_report_docs,
    get_history_and_meta,
    serialize_history,
)


_REPO_ROOT = Path(__file__).resolve().parents[5]
REPORTS_ROOT = _REPO_ROOT / "data" / "reports"


def _reports_cache_available() -> bool:
    """Check that at least one CFPR JSON artifact exists."""
    cfpr_dir = REPORTS_ROOT / "cfpr"
    if not cfpr_dir.is_dir():
        return False
    return len(list(cfpr_dir.glob("*.json"))) > 0


pytestmark = pytest.mark.skipif(not _reports_cache_available(), reason="data/reports/cfpr not populated")


class TestDocumentStoreRealArtifacts:
    """Smoke-test DocumentStore against real extracted CFPR reports."""

    def test_loads_all_cfpr_editions(self) -> None:
        """All cached CFPR editions load in chronological order."""
        store = DocumentStore({"cfpr": REPORTS_ROOT / "cfpr"})
        docs = store.list_docs("cfpr")
        assert len(docs) >= 2  # we have at least a few years
        ids = [d.meta.doc_id for d in docs]
        assert any(i.startswith("202") for i in ids)
        # Chronological order
        dates = [d.meta.publication_date for d in docs]
        assert dates == sorted(dates)

    def test_cutoff_filtering_excludes_future_reports(self) -> None:
        """Reports published after as_of must not appear."""
        store = DocumentStore({"cfpr": REPORTS_ROOT / "cfpr"})
        docs_far_past = store.list_docs("cfpr", as_of=date(2019, 1, 1))
        assert docs_far_past == []
        docs_all = store.list_docs("cfpr")
        docs_far_future = store.list_docs("cfpr", as_of=date(2030, 1, 1))
        assert len(docs_far_future) == len(docs_all)


class TestForecastContextWithDocs:
    """Verify ForecastContext.get_documents() end-to-end."""

    def test_get_documents_returns_cutoff_filtered(self) -> None:
        """get_documents returns only editions published on or before as_of."""
        store = DocumentStore({"cfpr": REPORTS_ROOT / "cfpr"})
        series_store = SeriesStore()
        ctx = ForecastContext(series_store, as_of=datetime(2023, 6, 1), doc_store=store)
        docs = ctx.get_documents("cfpr")
        assert all(d.meta.publication_date <= date(2023, 6, 1) for d in docs)
        assert len(docs) > 0

    def test_get_documents_empty_when_no_store(self) -> None:
        """Without a DocumentStore, document access returns empties."""
        ctx = ForecastContext(SeriesStore(), as_of=datetime(2023, 6, 1))
        assert ctx.get_documents("cfpr") == []
        assert ctx.document_sources == []


class TestPromptPipelineSmoke:
    """Exercise the full prompt-building pipeline (docs → preamble → prompt)."""

    def test_fetch_report_docs_with_real_data(self) -> None:
        """Real CFPR docs flow through fetch into a truncated preamble."""
        store = DocumentStore({"cfpr": REPORTS_ROOT / "cfpr"})
        ctx = ForecastContext(SeriesStore(), as_of=datetime(2023, 6, 1), doc_store=store)
        config = LLMPredictorConfig(report_sources=["cfpr"], report_max_chars=5000)
        docs = fetch_report_docs(config=config, context=ctx)
        assert len(docs) > 0

        preamble = build_report_preamble(docs, max_chars=5000)
        assert "Source: cfpr" in preamble
        assert "Published:" in preamble
        # Each doc should have been truncated to <= 5000 + small metadata
        # We can't assert exact chars, but we can assert existence
        assert len(preamble) < 5000 * len(docs) + 500 * len(docs)

    def test_full_prompt_construction_with_series(self) -> None:
        """Simulate what QuantileGridLLMPredictor does."""
        store = DocumentStore({"cfpr": REPORTS_ROOT / "cfpr"})
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=36, freq="MS"),
                "value": [100 + i * 0.5 for i in range(36)],
            }
        )
        series_store = SeriesStore()
        series_store.put(
            "food_cpi",
            df,
            SeriesMetadata(
                series_id="food_cpi",
                description="Food CPI (2002=100)",
                source="StatCan",
                units="Index",
                frequency="MS",
            ),
        )
        ctx = ForecastContext(series_store, as_of=datetime(2022, 7, 1), doc_store=store)

        task = ForecastingTask(
            task_id="food_cpi_test",
            description="Forecast monthly Food CPI values.",
            target_series_id="food_cpi",
            payload_type="continuous",
            frequency="MS",
            horizons=[1, 3, 6, 9, 12],
        )

        config = LLMPredictorConfig(report_sources=["cfpr"], report_max_chars=5000)
        report_docs = fetch_report_docs(config=config, context=ctx)
        preamble = build_report_preamble(report_docs, max_chars=5000)

        series_df, series_meta = get_history_and_meta(task, ctx)
        history_str = serialize_history(series_df, precision=2)

        # Simulate user prompt construction with preamble
        user_prompt = f"Task: {task.description}\n\nHistory:\n{history_str}"
        if preamble:
            user_prompt = (
                "You are provided with the following economic report(s) "
                "published before the forecast date. Use them as context "
                "for your forecast.\n\n" + preamble + "\n\n---\n\n" + user_prompt
            )

        # Sanity checks on the assembled prompt
        assert "Food CPI" in user_prompt
        assert "Source: cfpr" in user_prompt
        assert "History:" in user_prompt
        assert "---" in user_prompt
        # Preamble comes before history
        idx_hist = user_prompt.index("History:")
        idx_source = user_prompt.index("Source: cfpr")
        assert idx_source < idx_hist

        # Check that we have at least one report whose publication_date
        # matches the cutoff (should include 2021 report published Dec 2020)
        assert len(report_docs) >= 1


class TestPromptPipelineNoReportsFallback:
    """When report_sources is None, the prompt is unchanged."""

    def test_no_preamble_when_report_sources_is_none(self) -> None:
        """report_sources=None yields no docs and an empty preamble."""
        store = DocumentStore({"cfpr": REPORTS_ROOT / "cfpr"})
        ctx = ForecastContext(SeriesStore(), as_of=datetime(2023, 6, 1), doc_store=store)
        config = LLMPredictorConfig(report_sources=None)
        docs = fetch_report_docs(config=config, context=ctx)
        assert docs == []
        assert build_report_preamble(docs) == ""

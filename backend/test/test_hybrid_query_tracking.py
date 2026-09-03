import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from main.service.hybrid_query_service import (
    HybridQueryService,
    TrackingQueryEngine,
    ToolExecution,
)


class TrackingQueryEngineTests(unittest.TestCase):
    def test_aquery_records_success_and_delegates(self):
        inner = MagicMock()
        inner.aquery = AsyncMock(return_value="Rutgers University lecture notes")
        events: list[ToolExecution] = []
        engine = TrackingQueryEngine("bm25_search", inner, events)

        result = asyncio.run(engine.aquery("Which lecture notes by Zaborszky?"))

        self.assertEqual(result, "Rutgers University lecture notes")
        inner.aquery.assert_awaited_once()
        self.assertEqual(events[0].name, "bm25_search")
        self.assertEqual(events[0].status, "success")

    def test_aquery_records_error(self):
        inner = MagicMock()
        inner.aquery = AsyncMock(side_effect=RuntimeError("index unavailable"))
        events: list[ToolExecution] = []
        engine = TrackingQueryEngine("semantic_search", inner, events)

        with self.assertRaises(RuntimeError):
            asyncio.run(engine.aquery("anatomy of the somatosensory system"))

        self.assertEqual(events[0].status, "error")
        self.assertIn("index unavailable", events[0].detail)


class AgentRoutingTrackingTests(unittest.IsolatedAsyncioTestCase):
    async def test_hf_only_success_is_not_replaced_by_document_fallback(self):
        service = HybridQueryService()
        service._initialized = True
        service.initialization_errors = []

        async def run_agent(**kwargs):
            service.events.append(ToolExecution("model_search", "success"))
            return "Hugging Face model results"

        service.agent = MagicMock()
        service.agent.run = AsyncMock(side_effect=run_agent)
        service._fallback = AsyncMock()

        result = await service.execute("Find a medical model on Hugging Face")

        self.assertEqual(result.tools_used, ["model_search"])
        self.assertEqual(result.routing, "agent:model_search")
        service._fallback.assert_not_awaited()

    async def test_multi_tool_success_and_failure_are_reported_together(self):
        service = HybridQueryService()
        service._initialized = True
        service.initialization_errors = []

        async def run_agent(**kwargs):
            service.events.extend([
                ToolExecution("semantic_search", "success"),
                ToolExecution("model_search", "error", "HF temporarily unavailable"),
                ToolExecution("clinical_sql", "success"),
            ])
            return "Combined answer from the available sources"

        service.agent = MagicMock()
        service.agent.run = AsyncMock(side_effect=run_agent)

        result = await service.execute("Compare the textbook, clinical data, and Hugging Face")

        self.assertEqual(result.tools_used, ["semantic_search", "clinical_sql"])
        self.assertEqual(result.routing, "agent:semantic_search+clinical_sql")
        self.assertEqual(
            result.tool_errors,
            [{"tool": "model_search", "message": "HF temporarily unavailable"}],
        )


if __name__ == "__main__":
    unittest.main()

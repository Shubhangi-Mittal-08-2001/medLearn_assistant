import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel, ConfigDict

from main.service.hybrid_query_service import (
    HybridQueryService,
    TrackingTool,
    TrackingQueryEngine,
    ToolExecution,
)
from main.tools.custom_tools import custom_inference_tool


class RepoDetailsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_ids: list[str]


class RepoSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = None
    repo_types: list[str] = ["model", "dataset"]
    filters: list[str] | None = None
    sort: str | None = None
    limit: int = 20


class TrackingToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_hf_inference_uses_only_canonical_arguments(self):
        search_result = {"repo_id": "test-org/medical-classifier"}
        events = [ToolExecution("hub_repo_search", "success")]
        inner = MagicMock()
        inner.metadata = custom_inference_tool.metadata
        inner.acall = AsyncMock(return_value='[{"label": "medical", "score": 0.9}]')
        inference = TrackingTool(inner, events)

        await inference.acall(
            model_id=search_result["repo_id"],
            text="The patient has severe chest pain.",
        )

        inner.acall.assert_awaited_once_with(
            model_id="test-org/medical-classifier",
            text="The patient has severe chest pain.",
        )
        self.assertEqual([event.name for event in events], ["hub_repo_search", "hf_inference"])

    async def test_hf_inference_rejects_guessed_argument_names(self):
        for wrong_name in ("model", "repo_id", "engine"):
            events = [ToolExecution("hub_repo_search", "success")]
            inner = MagicMock()
            inner.metadata = custom_inference_tool.metadata
            inner.acall = AsyncMock(return_value="should not run")

            with self.assertRaisesRegex(ValueError, "expected.*model_id.*text"):
                await TrackingTool(inner, events).acall(
                    **{wrong_name: "test-org/medical-classifier", "text": "chest pain"}
                )

            inner.acall.assert_not_awaited()

    async def test_hub_repo_details_requires_repo_ids_array(self):
        inner = MagicMock()
        inner.metadata.name = "hub_repo_details"
        inner.metadata.fn_schema = RepoDetailsInput
        inner.acall = AsyncMock(return_value={"id": "test-org/medical-classifier"})
        tool = TrackingTool(inner, [])

        await tool.acall(repo_ids=["test-org/medical-classifier"])
        inner.acall.assert_awaited_once_with(repo_ids=["test-org/medical-classifier"])

        for invalid in ({"repo_id": "test-org/medical-classifier"}, {}, {"repo_ids": []}):
            inner.acall.reset_mock()
            with self.assertRaises(ValueError):
                await tool.acall(**invalid)
            inner.acall.assert_not_awaited()

    async def test_hub_repo_search_rejects_non_schema_filter_names(self):
        inner = MagicMock()
        inner.metadata.name = "hub_repo_search"
        inner.metadata.fn_schema = RepoSearchInput
        inner.acall = AsyncMock(return_value={"repo_id": "test-org/medical-classifier"})
        tool = TrackingTool(inner, [])

        await tool.acall(
            query="biomedical classification",
            repo_types=["model"],
            filters=["text-classification", "language:en"],
            sort="downloads",
            limit=10,
        )
        inner.acall.assert_awaited_once()

        inner.acall.reset_mock()
        with self.assertRaisesRegex(ValueError, "pipeline_tag"):
            await tool.acall(query="biomedical", pipeline_tag="text-classification")
        inner.acall.assert_not_awaited()

    async def test_inference_is_blocked_after_failed_search(self):
        events = [ToolExecution("hub_repo_search", "error", "connection failed")]
        inference = MagicMock()
        inference.metadata.name = "hf_inference"
        inference.acall = AsyncMock(return_value="should not run")

        with self.assertRaisesRegex(RuntimeError, "requires a successful hub_repo_search"):
            await TrackingTool(inference, events).acall(
                model_id="invented/model", text="The patient has severe chest pain."
            )

        inference.acall.assert_not_awaited()
        self.assertEqual(events[-1].name, "hf_inference")
        self.assertEqual(events[-1].status, "error")

    async def test_exception_group_records_leaf_error_only(self):
        inner = MagicMock()
        inner.metadata.name = "hub_repo_search"
        inner.metadata.fn_schema = RepoSearchInput
        inner.acall = AsyncMock(side_effect=ExceptionGroup(
            "TaskGroup failure", [RuntimeError("actual MCP connection failure")]
        ))
        events: list[ToolExecution] = []

        with self.assertRaises(ExceptionGroup):
            await TrackingTool(inner, events).acall(query="biomedical model")

        self.assertEqual(events[0].status, "error")
        self.assertIn("actual MCP connection failure", events[0].detail)
        self.assertNotIn("TaskGroup failure", events[0].detail)

    async def test_mcp_error_result_is_not_recorded_as_success(self):
        raw = MagicMock()
        raw.isError = True
        raw.content = [MagicMock(text="invalid max_results")]
        response = MagicMock(raw_output=raw)
        inner = MagicMock()
        inner.metadata.name = "hub_repo_search"
        inner.metadata.fn_schema = RepoSearchInput
        inner.acall = AsyncMock(return_value=response)
        events: list[ToolExecution] = []

        with self.assertRaises(RuntimeError):
            await TrackingTool(inner, events).acall(query="biomedical model")

        self.assertEqual([(event.name, event.status) for event in events], [("hub_repo_search", "error")])
        self.assertIn("invalid max_results", events[0].detail)


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
    async def test_hf_search_then_inference_success(self):
        service = HybridQueryService()
        service._initialized = True
        service.initialization_errors = []
        call_order: list[str] = []

        async def run_agent(**kwargs):
            call_order.append("hub_repo_search")
            service.events.append(ToolExecution("hub_repo_search", "success"))
            call_order.append("hf_inference")
            service.events.append(ToolExecution("hf_inference", "success"))
            return '[{"label": "LABEL_1", "score": 0.99}]'

        service.agent = MagicMock()
        service.agent.run = AsyncMock(side_effect=run_agent)

        result = await service.execute(
            'Find a biomedical text classification model and classify: "The patient has severe chest pain."'
        )

        self.assertEqual(call_order, ["hub_repo_search", "hf_inference"])
        self.assertEqual(result.tools_used, ["hub_repo_search", "hf_inference"])
        self.assertEqual(result.tool_errors, [])

    async def test_hf_search_failure_does_not_call_inference(self):
        service = HybridQueryService()
        service._initialized = True
        service.initialization_errors = []
        inference = AsyncMock()

        async def run_agent(**kwargs):
            try:
                raise RuntimeError("actual MCP connection failure")
            except RuntimeError as exc:
                service.events.append(ToolExecution("hub_repo_search", "error", str(exc)))
            return "Hugging Face model search failed; inference was not run."

        service.agent = MagicMock()
        service.agent.run = AsyncMock(side_effect=run_agent)
        service._fallback = AsyncMock()

        result = await service.execute("Find and run a biomedical classifier")

        inference.assert_not_awaited()
        service._fallback.assert_not_awaited()
        self.assertEqual(result.tools_used, [])
        self.assertEqual(result.tool_errors, [{"tool": "hub_repo_search", "message": "actual MCP connection failure"}])
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

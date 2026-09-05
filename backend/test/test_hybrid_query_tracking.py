import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch
from pydantic import BaseModel, ConfigDict

from main.service.hybrid_query_service import (
    HybridQueryResult,
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


class QueryEngineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str


class TrackingToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_document_tool_normalizes_query_and_ignores_top_k(self):
        inner = MagicMock()
        inner.metadata.name = "hybrid_document_search"
        inner.metadata.fn_schema = QueryEngineInput
        inner.acall = AsyncMock(return_value="document evidence")
        tool = TrackingTool(inner, [])

        result = await tool.acall(query="long-term GLP-1 effects", top_k=5)

        self.assertEqual(result, "document evidence")
        inner.acall.assert_awaited_once_with(input="long-term GLP-1 effects")

    async def test_rag_evaluation_flushes_confidence_score(self):
        service = HybridQueryService()
        service.source_nodes[:] = [{"text": "document evidence"}]
        langfuse = MagicMock()

        with patch(
            "main.service.hybrid_query_service.get_langfuse_client",
            return_value=langfuse,
        ), patch(
            "main.service.hybrid_query_service.evaluate_faithfulness_score",
            new=AsyncMock(return_value=0.8),
        ), patch(
            "main.service.hybrid_query_service.evaluate_answer_relevance",
            new=AsyncMock(return_value=0.6),
        ):
            result = await service._evaluate_rag_evidence("question", "answer")

        self.assertEqual(result, (0.8, 0.6, 0.7))
        langfuse.create_score.assert_called_once_with(
            trace_id="hybrid-query-evaluation",
            name="confidence",
            value=0.7,
            comment="RAG confidence derived from faithfulness and answer relevance: 0.70",
        )
        langfuse.flush.assert_called_once_with()

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
    def test_medical_model_selection_rejects_irrelevant_language_and_gguf(self):
        candidates = [
            "mradermacher/ZamAI-Sentiment-Pashto-GGUF",
            "tasal9/ZamAI-Sentiment-Pashto",
            "medical-org/clinical-symptom-classifier",
        ]

        selected = HybridQueryService._best_medical_model_id(candidates)

        self.assertEqual(selected, "medical-org/clinical-symptom-classifier")

    async def test_explicit_classification_continues_from_search_to_inference(self):
        service = HybridQueryService()
        service.events[:] = [ToolExecution("hub_repo_search", "success")]
        service.external_evidence[:] = [{
            "tool": "hub_repo_search",
            "content": '{"id":"medical-org/chest-pain-classifier"}',
        }]
        inference = MagicMock()
        inference.metadata.name = "hf_inference"

        async def run_inference(**kwargs):
            service.events.append(ToolExecution("hf_inference", "success"))
            return '[{"label":"urgent","score":0.97}]'

        inference.acall = AsyncMock(side_effect=run_inference)
        service.tools = [inference]

        answer = await service._complete_requested_hf_inference(
            "Find a biomedical text classification model and use it to classify this sentence The patient has severe chest pain",
            "Search-only answer",
        )

        inference.acall.assert_awaited_once_with(
            model_id="medical-org/chest-pain-classifier",
            text="The patient has severe chest pain",
        )
        self.assertIn("Classification result", answer)
        self.assertEqual(service.events[-1], ToolExecution("hf_inference", "success"))

    async def test_unsupported_provider_candidate_retries_next_medical_model(self):
        service = HybridQueryService()
        service.events[:] = [ToolExecution("hub_repo_search", "success")]
        service.external_evidence[:] = [{
            "tool": "hub_repo_search",
            "content": (
                '{"id":"medical-org/unsupported-clinical-model"},'
                '{"id":"biomed-org/hosted-classifier"}'
            ),
        }]
        inference = MagicMock()
        inference.metadata.name = "hf_inference"
        inference.acall = AsyncMock(side_effect=[
            RuntimeError("Model not supported by provider hf-inference"),
            '[{"label":"urgent","score":0.97}]',
        ])
        service.tools = [inference]

        answer = await service._complete_requested_hf_inference(
            "Find a biomedical model and classify this sentence The patient has severe chest pain",
            "search answer",
        )

        self.assertEqual(inference.acall.await_count, 2)
        self.assertIn("medical-org/unsupported-clinical-model", answer)
        self.assertIn("Classification result", answer)

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


class SourceAwareEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_hybrid_faithfulness_redirects_to_hugging_face_mcp(self):
        service = HybridQueryService()
        hf_tool = MagicMock()
        hf_tool.metadata.name = "paper_search"
        service.tools = [hf_tool]
        service.tool_origins = {"paper_search": "huggingface_mcp"}
        service.tool_sources = {"paper_search": "mcp"}
        service.events[:] = [ToolExecution("hybrid_document_search", "success")]
        initial = HybridQueryResult(
            "question", "weak local answer", [], ["hybrid_document_search"], [],
            "agent:hybrid_document_search", {"faithfulness": 0.4},
        )

        async def run_huggingface_agent(**kwargs):
            service.events.append(ToolExecution("paper_search", "success"))
            service.external_evidence.append({"tool": "paper_search", "content": "source:123"})
            return "Externally supported answer (PMID:123)"

        agent = MagicMock()
        agent.run = AsyncMock(side_effect=run_huggingface_agent)
        service._create_openai_agent = MagicMock(return_value=agent)
        service._evaluate_result = AsyncMock(return_value={"faithfulness": 0.8, "mode": "rag+mcp"})

        result = await service._redirect_weak_rag_to_huggingface_mcp("question", initial)

        service._create_openai_agent.assert_called_once_with([hf_tool])
        self.assertIn("Externally supported", result.answer)
        self.assertEqual(result.tools_used, ["hybrid_document_search", "paper_search"])

    async def test_low_faithfulness_uses_hugging_face_mcp(self):
        service = HybridQueryService()
        hf_tool = MagicMock()
        hf_tool.metadata.name = "hub_repo_search"
        service.tools = [hf_tool]
        service.tool_origins = {"hub_repo_search": "huggingface_mcp"}
        initial = HybridQueryResult(
            "question", "weak local answer", [], ["hybrid_document_search"], [],
            "agent:hybrid_document_search", {"faithfulness": 0.2},
        )
        async def run_huggingface_agent(**kwargs):
            service.events.append(ToolExecution("hub_repo_search", "success"))
            return "Insufficient Hugging Face evidence"

        agent = MagicMock()
        agent.run = AsyncMock(side_effect=run_huggingface_agent)
        service._create_openai_agent = MagicMock(return_value=agent)
        service._evaluate_result = AsyncMock(return_value={"faithfulness": 0.2, "mode": "rag+mcp"})

        result = await service._redirect_weak_rag_to_huggingface_mcp("question", initial)

        self.assertEqual(result.tools_used, ["hub_repo_search"])
        service._create_openai_agent.assert_called_once_with([hf_tool])

    async def test_each_document_tool_uses_only_rag_evaluation(self):
        for tool_name in ("semantic_search", "bm25_search", "hybrid_document_search"):
            with self.subTest(tool=tool_name):
                service = HybridQueryService()
                service.source_nodes[:] = [{"text": "document evidence"}]
                service._evaluate_rag_evidence = AsyncMock(return_value=(0.8, 0.6, 0.7))
                service._evaluate_mcp_evidence = AsyncMock()

                evaluation = await service._evaluate_result("question", "answer", [tool_name])

                self.assertEqual(evaluation["mode"], "rag")
                self.assertEqual(evaluation["faithfulness"], 0.8)
                self.assertEqual(evaluation["relevance"], 0.6)
                self.assertEqual(evaluation["rag_confidence"], 0.7)
                self.assertEqual(evaluation["overall_confidence"], 0.7)
                service._evaluate_rag_evidence.assert_awaited_once_with("question", "answer")
                service._evaluate_mcp_evidence.assert_not_awaited()

    async def test_mcp_only_skips_rag_metrics(self):
        service = HybridQueryService()
        service._evaluate_rag_evidence = AsyncMock()
        service._evaluate_mcp_evidence = AsyncMock(return_value=0.86)

        evaluation = await service._evaluate_result(
            "question", "answer", ["hub_repo_search", "hf_inference"]
        )

        self.assertEqual(evaluation["mode"], "mcp")
        self.assertIsNone(evaluation["faithfulness"])
        self.assertIsNone(evaluation["relevance"])
        self.assertEqual(evaluation["overall_confidence"], 0.86)
        service._evaluate_rag_evidence.assert_not_awaited()

    async def test_mixed_evaluates_sources_independently(self):
        service = HybridQueryService()
        service._evaluate_rag_evidence = AsyncMock(return_value=(0.9, 0.8, 0.85))
        service._evaluate_mcp_evidence = AsyncMock(return_value=0.95)

        evaluation = await service._evaluate_result(
            "question", "answer", ["semantic_search", "hub_repo_search", "hf_inference"]
        )

        self.assertEqual(evaluation["mode"], "rag+mcp")
        self.assertEqual(evaluation["rag_confidence"], 0.85)
        self.assertEqual(evaluation["mcp_confidence"], 0.95)
        self.assertAlmostEqual(evaluation["overall_confidence"], 0.9)

    async def test_sql_and_none_do_not_run_evaluators(self):
        for tools_used, expected_mode in ((["clinical_sql"], "sql"), ([], "none")):
            service = HybridQueryService()
            service._evaluate_rag_evidence = AsyncMock()
            service._evaluate_mcp_evidence = AsyncMock()

            evaluation = await service._evaluate_result("question", "answer", tools_used)

            self.assertEqual(evaluation["mode"], expected_mode)
            self.assertIsNone(evaluation["overall_confidence"])
            service._evaluate_rag_evidence.assert_not_awaited()
            service._evaluate_mcp_evidence.assert_not_awaited()

    async def test_failed_inference_is_not_usable_evidence(self):
        service = HybridQueryService()
        service.events[:] = [
            ToolExecution("hub_repo_search", "success"),
            ToolExecution("hf_inference", "error", "inference failed"),
        ]
        service.external_evidence[:] = [
            {"tool": "hub_repo_search", "content": "model search result"},
            {"tool": "hf_inference", "content": "stale result"},
        ]
        confidence = AsyncMock(return_value={"confidence": 80})

        from unittest.mock import patch
        with patch("main.service.hybrid_query_service.evaluate_confidence_score", confidence):
            score = await service._evaluate_mcp_evidence("question", "answer")

        self.assertEqual(score, 0.8)
        evidence = confidence.await_args.args[2]
        self.assertEqual([item["tool"] for item in evidence], ["hub_repo_search"])

    async def test_execute_clears_both_evidence_collections(self):
        service = HybridQueryService()
        service._initialized = True
        service.initialization_errors = []
        service.source_nodes[:] = [{"text": "old RAG"}]
        service.external_evidence[:] = [{"tool": "hf_inference", "content": "old MCP"}]
        service.agent = MagicMock()
        service.agent.run = AsyncMock(return_value="answer")
        service._fallback = AsyncMock(return_value=MagicMock())

        await service.execute("new question")

        self.assertEqual(service.source_nodes, [])
        self.assertEqual(service.external_evidence, [])

    async def test_permission_failure_returns_truthful_non_handoff_answer(self):
        service = HybridQueryService()
        service.events[:] = [ToolExecution("hub_repo_search", "success")]
        service.external_evidence[:] = [{
            "tool": "hub_repo_search",
            "content": '{"id":"medical-org/clinical-classifier"}',
        }]
        inference = MagicMock()
        inference.metadata.name = "hf_inference"
        inference.acall = AsyncMock(side_effect=RuntimeError("403 Forbidden: insufficient permissions"))
        service.tools = [inference]

        answer = await service._complete_requested_hf_inference(
            "Find a biomedical model and classify this sentence The patient has severe chest pain",
            "agent stopped after search",
        )

        self.assertIn("rule-based fallback", answer)
        self.assertIn("not a Hugging Face model output", answer)
        self.assertIn("potential emergency", answer)
        self.assertNotIn("escalated", answer)

    async def test_no_matching_model_returns_direct_classification_not_options(self):
        service = HybridQueryService()
        service.events[:] = [ToolExecution("hub_repo_search", "success")]
        service.external_evidence[:] = [{
            "tool": "hub_repo_search",
            "content": '{"id":"tasal9/ZamAI-Sentiment-Pashto"}',
        }]

        answer = await service._complete_requested_hf_inference(
            "Find a biomedical model and classify this sentence The patient has severe chest pain",
            "Which option do you want?",
        )

        self.assertIn("Symptom: chest pain", answer)
        self.assertIn("Severity: severe", answer)
        self.assertNotIn("Which option", answer)


if __name__ == "__main__":
    unittest.main()

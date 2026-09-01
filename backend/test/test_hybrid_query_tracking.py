import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from main.service.hybrid_query_service import TrackingQueryEngine, ToolExecution


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


if __name__ == "__main__":
    unittest.main()

import pytest

from main.service.query_router import QueryRouter, QueryType


@pytest.mark.parametrize(
    ("query", "expected_type", "expected_retrievers"),
    [
        (
            "How does tactile perception adapt as people grow older?",
            QueryType.VECTOR_ONLY,
            ["vector"],
        ),
        (
            "Define the exact temperature range for a nociceptor response.",
            QueryType.BM25_ONLY,
            ["bm25"],
        ),
        (
            "Compare Pacinian versus Meissner corpuscles.",
            QueryType.FUSION,
            ["vector", "bm25"],
        ),
    ],
)
def test_router_selects_expected_retrievers(query, expected_type, expected_retrievers):
    analysis = QueryRouter.analyze(query)

    assert analysis.query_type is expected_type
    assert analysis.recommended_retrievers == expected_retrievers
    assert 0 <= analysis.confidence <= 1
    assert analysis.reasons


def test_router_defaults_unknown_language_to_fusion():
    analysis = QueryRouter.analyze("xyzzy plugh")

    assert analysis.query_type is QueryType.FUSION
    assert analysis.confidence == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("query", "is_relevant"),
    [
        ("How do Pacinian receptors detect vibration?", True),
        ("What is the best paint color for a kitchen?", False),
    ],
)
def test_medlearn_scope_gate(query, is_relevant):
    assert QueryRouter.is_relevant_to_medlearn_documents(query) is is_relevant


def test_long_unclassified_query_records_complexity_reason():
    analysis = QueryRouter.analyze(" ".join(["unknown"] * 16))

    assert analysis.query_type is QueryType.FUSION
    assert any("Complex query" in reason for reason in analysis.reasons)


def test_routing_decision_is_logged(caplog):
    analysis = QueryRouter.analyze("Explain tactile perception")

    with caplog.at_level("INFO"):
        QueryRouter.log_routing_decision("Explain tactile perception", analysis)

    assert "query.routing" in caplog.text
    assert "retrievers=vector" in caplog.text

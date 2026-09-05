import pytest
from pydantic import ValidationError

from main.schema.models import QueryRequest, QueryResponse


def test_query_request_trims_user_input():
    request = QueryRequest(
        question="  How do Pacinian corpuscles detect vibration?  ",
        user_email="  learner@example.com  ",
    )

    assert request.question == "How do Pacinian corpuscles detect vibration?"
    assert request.user_email == "learner@example.com"


@pytest.mark.parametrize("email", ["", "missing-at-sign", "@example.com", "user@"])
def test_query_request_rejects_invalid_email(email):
    with pytest.raises(ValidationError):
        QueryRequest(question="Explain tactile sensation", user_email=email)


@pytest.mark.parametrize("question", ["", "   "])
def test_query_request_rejects_empty_question(question):
    with pytest.raises(ValidationError):
        QueryRequest(question=question, user_email="learner@example.com")


def test_query_response_has_safe_collection_defaults():
    first = QueryResponse(answer="one")
    second = QueryResponse(answer="two")

    first.tools_used.append("semantic_search")

    assert second.tools_used == []
    assert second.source_nodes == []
    assert second.tool_errors == []


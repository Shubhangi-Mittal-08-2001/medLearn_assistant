from main.service.query_router import QueryRouter


def test_housing_query_is_outside_medlearn_scope():
    assert not QueryRouter.is_relevant_to_medlearn_documents(
        "How do I find houses in Bangalore?"
    )


def test_somatosensory_query_is_in_medlearn_scope():
    assert QueryRouter.is_relevant_to_medlearn_documents(
        "How do Pacinian receptors detect vibration?"
    )

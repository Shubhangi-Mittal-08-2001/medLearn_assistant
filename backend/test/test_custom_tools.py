import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from main.tools.custom_tools import _call_with_hub_client, hf_inference_api


def test_hf_inference_uses_explicit_hugging_face_provider(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    client = MagicMock()
    client.text_classification.return_value = [SimpleNamespace(label="LABEL_1", score=0.9)]

    with patch("main.tools.custom_tools.InferenceClient", return_value=client) as constructor:
        result = json.loads(_call_with_hub_client(
            "test-org/medical-classifier", "The patient has severe chest pain."
        ))

    constructor.assert_called_once_with(token="test-token", provider="hf-inference")
    client.text_classification.assert_called_once_with(
        "The patient has severe chest pain.", model="test-org/medical-classifier"
    )
    assert result == [{"label": "LABEL_1", "score": 0.9}]


@pytest.mark.parametrize("model_id,text,error", [
    ("", "chest pain", "model_id is required"),
    ("test-org/model", "  ", "text is required"),
])
def test_hf_inference_validates_required_values(model_id, text, error):
    with pytest.raises(ValueError, match=error):
        hf_inference_api(model_id, text)

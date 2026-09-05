# MedLearn Assistant Backend

## Query pipeline

`POST /query` now follows one path:

1. Pydantic request validation
2. Guardrails input validation
3. Query classification
4. Hybrid agent invocation
5. Actual tool execution tracking (SQL / vector / BM25 / fusion / optional MCP)
6. Final LLM response
7. Output Guardrails + DetectPII redaction
8. Langfuse trace (when configured)
9. Faithfulness + relevance evaluation
10. Confidence + explicit human-help evaluation
11. Human handoff decision
12. Normal answer or handoff reference + background email

## Setup (PowerShell)

```powershell
cd "A:\Agentic AI\course3-project\medlearn-assistant\backend"
Copy-Item .env.example .env
# Fill your real keys/passwords in .env

Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
Remove-Item -Force uv.lock -ErrorAction SilentlyContinue
uv sync --python 3.12
.\.venv\Scripts\Activate.ps1
python -m guardrails_ai.detect_pii.post_install
uv run main.py
```

Open `http://127.0.0.1:8000/docs`.

## Example query body

```json
{
  "question": "Explain the difference between Pacinian and Meissner corpuscles",
  "user_email": "your-email@example.com"
}
```

The response includes `tools_used`, `tool_errors`, `classification`, Guardrails results,
evaluation scores, handoff status, and trace ID.

> `.env` is intentionally not included in the packaged backend. Keep secrets local.

## MCP integrations

- `HF_MCP_URL` exposes Hugging Face MCP tools; set `HF_TOKEN` to enable it.
  This is the application's only external MCP integration.

Hub search neither retrieves medical literature nor executes a model. The separate
`hf_inference` tool uses Inference Providers. Its fine-grained token must include
**Make calls to Inference Providers**; otherwise inference returns HTTP 403 even when
MCP discovery succeeds. `HF_INFERENCE_PROVIDER` defaults to `auto`.

An unavailable MCP server is recorded in `tool_errors`; SQL and local RAG remain usable.
When local document evidence scores below `RAG_FAITHFULNESS_THRESHOLD` or
`RAG_RELEVANCE_THRESHOLD` (both default to `0.70`), the service re-plans with the
configured Hugging Face MCP tools. It only accepts tools that can retrieve relevant
external content; a model/repository listing alone is not presented as medical evidence.

## Supported upload formats

Set the application upload configuration to:

```env
MAX_UPLOAD_SIZE_KB=10240
ALLOWED_FILE_TYPES=.pdf,.docx,.doc,.txt,.xlsx
IMAGES_FOLDER=stored_images
```

Processing behavior:
- `.pdf`: PyMuPDF text extraction + PDF image extraction/captioning.
- `.docx`: local Word paragraph/table extraction.
- `.doc`: Legacy format (requires conversion to .docx).
- `.txt`: local text parsing with encoding fallbacks.
- `.xlsx`: local worksheet/table extraction using openpyxl.


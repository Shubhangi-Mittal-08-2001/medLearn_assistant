# MedLearn Backend Architecture

## Ingestion

`POST /upload` validates filename, extension and size, then processes PDF, DOC/DOCX,
TXT or XLSX content. Text is semantically chunked into nodes. Markdown tables become
table-summary nodes. Embedded PDF images are extracted with PyMuPDF and captioned; the
caption is stored as an image-caption node. All nodes carry metadata and are embedded
into PostgreSQL/pgvector. Existing structured clinical tables are separate from the
vector collection and can be loaded directly into PostgreSQL.

## Query pipeline

`POST /query` -> Pydantic validation -> input guardrails -> request-scoped agent ->
agent-selected tool(s) -> final answer -> output guardrails/PII sanitization -> Langfuse
trace -> faithfulness/relevance/confidence evaluation -> optional human handoff.

### Agent tools

- `clinical_sql`: read-only natural-language querying of configured clinical tables.
  The user receives the factual result, not generated SQL.
- `semantic_search`: meaning-based vector similarity search over indexed nodes.
- `bm25_search`: keyword retrieval over indexed document nodes.
- `hybrid_document_search`: semantic + BM25 fusion/reranking.
- Hugging Face MCP tools: external model discovery/capabilities.
- `hf_inference`: executes a specific Hugging Face model after model selection; it is
  not a document/literature retriever.
- Optional medical MCP tools: configure `HF_MCP_URL` (and optionally
  `HF_TOKEN`) for a bioHugging Face MCP server such as a PubMed-capable
  server. The agent is prompted to use local documents first and only use a suitable
  medical external retrieval tool when local evidence is insufficient.

### Safety and metadata

Input guardrails detect empty input, common prompt-injection/jailbreak patterns, direct
harm/kill instruction requests, database mutation requests, and basic PII metadata.
Database execution is independently constrained to read-only statements. Output
Guardrails DetectPII/Presidio redacts PII and blocks raw SQL exposure. API source nodes
include the actual `retrieval_method` that produced them, such as `semantic_search`,
`bm25_search`, or `hybrid_document_search`.

### Evaluation, tracing, handoff

Langfuse records query-pipeline metadata including routing, selected tools, safety flags
and prompt version. Faithfulness compares the final answer with actual tool evidence;
relevance compares the answer with the question; confidence is a separate LLM score.
Low/absent evidence, low evaluation scores, low confidence combined with quality risk,
or an explicit request for human help can trigger handoff. Handoff includes a reference
ID, trace ID, scores, source/tool context and email notification.

## Medical knowledge fallback

For a question such as "What is the structure of the eye?", the agent should first use
local document retrieval. If the returned evidence is missing or insufficient, it may
call a configured medical/biomedical MCP retrieval tool. Hugging Face `model_search`
alone is not treated as medical evidence, and `hf_inference` is not used as a replacement
for reliable anatomy/literature retrieval. If no reliable evidence source is available,
the answer should state the limitation and allow the handoff policy to decide escalation.

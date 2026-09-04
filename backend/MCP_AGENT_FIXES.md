# MedLearn MCP / Evidence Retry Fixes

Updated prompt version: `medlearn-agent-v2.6`

## Fixed

1. Tool-returned text is no longer automatically treated as good evidence.
2. The orchestration layer now uses an LLM evidence gate for `RELEVANT` + `SUFFICIENT`.
3. If evidence is irrelevant/insufficient, the same request-scoped agent gets another reasoning turn and may choose a different available tool.
4. No fixed semantic -> BM25 -> hybrid -> MCP sequence is imposed.
5. Hybrid retrieval is recognized as already containing semantic + BM25 retrieval.
6. Hugging Face MCP wording was updated from the old Sprint-8 `model_search` name to the currently discovered Hub search capability such as `hub_repo_search`.
7. Hugging Face MCP now logs `status=not_configured` when `HF_MCP_URL` is missing.
8. Public confidence is capped by faithfulness/relevance so a relevance score of 0 can no longer coexist with 95 confidence.

## Expected pupil-query behavior

If local hybrid evidence is unrelated to the pupil:

- evidence gate => RELEVANT: NO / SUFFICIENT: NO
- agent gets another tool-selection turn
- if a Hugging Face MCP tool is registered, the agent may choose it
- if no suitable Hugging Face retrieval tool is configured, the request ends in handoff rather than using unrelated local evidence

## Important MCP distinction

The Hugging Face MCP server can load tools such as `hub_repo_search`, `hub_repo_details`, `hf_fs`, and `hf_whoami`. These are Hugging Face Hub capabilities, not general medical literature search.

The runtime uses the configured Hugging Face MCP server only. When it exposes no
tool capable of retrieving relevant external content, the system does not treat a
repository/model listing as medical evidence and instead permits handoff.

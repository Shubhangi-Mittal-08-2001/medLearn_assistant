"""
LLM-as-a-Judge Evaluators and Evaluation Pipeline
Implements faithfulness and answer relevance evaluators, plus end-to-end evaluation pipeline.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langfuse import Langfuse
from llama_index.core.settings import Settings

from main.service.rag_service import initialize_services, get_query_engine, add_documents_to_index, get_embed_model
from main.service.indexing import load_documents
from main.evaluation.dataset import get_langfuse_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

DATASET_NAME = "product_manual"


async def evaluate_faithfulness_score(
    langfuse,
    trace_id: str,
    query: str,
    context: str,
    answer: str,
) -> float:
    """
    Evaluate if answer is grounded in retrieved context (0.0 - 1.0).
    Uses Settings.llm (Azure OpenAI configured in initialize_services).
    """

    judge_prompt = f"""
You are evaluating a RAG system's answer for faithfulness.

Question: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Task: Rate faithfulness from 0.0 to 1.0:
- 1.0 = Every claim in the answer is explicitly supported by the context
- 0.7-0.9 = Most claims supported, minor unsupported details
- 0.4-0.6 = Mix of supported and unsupported claims
- 0.1-0.3 = Few claims supported, significant fabrication
- 0.0 = Answer contradicts context or is entirely invented

Provide step-by-step reasoning, then give your score.

Format:
Reasoning: [Identify each claim and verify against context]
Score: [0.0 to 1.0]
"""

    try:
        # Use the same LLM your RAG uses (AzureOpenAI via LlamaIndex Settings)
        response = Settings.llm.complete(judge_prompt)
        judgment = response.text

        # Extract line with "Score:"
        score_line = [line for line in judgment.split("\n") if "Score:" in line][0]
        score_str = score_line.split(":", 1)[-1].strip()
        score = float(score_str)

        # Clamp range
        score = max(0.0, min(1.0, score))

        # Attach score to Langfuse using create_score
        langfuse.create_score(
            trace_id=trace_id,
            name="faithfulness",
            value=score,
            comment=judgment,
        )

        logger.info(f"Faithfulness score for trace {trace_id}: {score:.2f}")
        return score

    except Exception as e:
        logger.error(f"Faithfulness evaluation error: {e}", exc_info=True)
        return 0.0


async def evaluate_answer_relevance(
    langfuse,
    trace_id: str,
    query: str,
    answer: str,
) -> float:
    """
    Evaluate how well the answer addresses the question (0.0 - 1.0).
    """

    judge_prompt = f"""
You are evaluating how well an answer addresses a user's question.

Question: {query}

Answer: {answer}

Task: Rate answer relevance from 0.0 to 1.0:
- 1.0 = Directly and completely answers the question
- 0.7-0.9 = Answers the question but includes some extra or tangential info
- 0.4-0.6 = Partially relevant, missing key points
- 0.1-0.3 = Mostly irrelevant to the question
- 0.0 = Completely unrelated

Provide reasoning, then score.

Format:
Reasoning: [explain relevance]
Score: [0.0 to 1.0]
"""

    try:
        response = Settings.llm.complete(judge_prompt)
        judgment = response.text

        score_line = [line for line in judgment.split("\n") if "Score:" in line][0]
        score_str = score_line.split(":", 1)[-1].strip()
        score = float(score_str)

        score = max(0.0, min(1.0, score))

        langfuse.create_score(
            trace_id=trace_id,
            name="answer_relevance",
            value=score,
            comment=judgment,
        )

        logger.info(f"Answer relevance score for trace {trace_id}: {score:.2f}")
        return score

    except Exception as e:
        logger.error(f"Answer relevance evaluation error: {e}", exc_info=True)
        return 0.0


async def query_with_evaluation(
    langfuse: Langfuse,
    question: str,
    expected_output: Optional[str] = None,
    similarity_top_k: int = 2,
) -> Dict[str, Any]:
    """
    Execute a RAG query using your existing query engine,
    then trigger automated evaluation (faithfulness + answer relevance).
    """

    # Create a span in Langfuse for this dataset item
    span = langfuse.start_span(
        name="rag_query_evaluation",
        input={"question": question, "expected_output": expected_output},
        metadata={"similarity_top_k": similarity_top_k, "dataset": DATASET_NAME},
    )
    trace_id = span.trace_id

    logger.info("\n" + "=" * 60)
    logger.info(f"Question: {question}")
    logger.info(f"Trace ID: {trace_id}")
    logger.info("=" * 60)

    # Use your existing query engine
    query_engine = get_query_engine(similarity_top_k=similarity_top_k)
    response = query_engine.query(question)
    answer = str(response)

    # Build context from source_nodes
    context_chunks: List[str] = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for node in response.source_nodes:
            try:
                context_chunks.append(node.text)
            except Exception:
                continue

    context = "\n\n".join(context_chunks)

    logger.info(f"Answer: {answer}")
    logger.info(f"Retrieved {len(context_chunks)} context chunks")

    # Run evaluations (LLM-as-a-judge) asynchronously
    faithfulness_task = asyncio.create_task(
        evaluate_faithfulness_score(langfuse, trace_id, question, context, answer)
    )
    relevance_task = asyncio.create_task(
        evaluate_answer_relevance(langfuse, trace_id, question, answer)
    )

    faithfulness_score = await faithfulness_task
    relevance_score = await relevance_task

    logger.info("\n" + "-" * 60)
    logger.info("Evaluation Results:")
    logger.info(f"  Faithfulness:     {faithfulness_score:.2f}")
    logger.info(f"  Answer Relevance: {relevance_score:.2f}")
    logger.info("-" * 60)

    # Attach output to the span and end it
    span.update(output={"answer": answer})
    span.end()

    return {
        "question": question,
        "expected_output": expected_output,
        "answer": answer,
        "trace_id": trace_id,
        "faithfulness": faithfulness_score,
        "answer_relevance": relevance_score,
    }


async def run_dataset_evaluation():
    """
    Evaluation Pipeline:
    - Initialize RAG services
    - Load dataset from Langfuse
    - For each item, run query + automated evaluation
    - Print aggregate metrics
    """
    # 1. Initialize RAG backend
    initialize_services()
    
    # Load and index documents if needed
    try:
        embed_model = get_embed_model()
        documents = load_documents()
        add_documents_to_index(documents)
        logger.info("✓ Documents indexed")
    except Exception as e:
        logger.warning(f"Could not index documents: {e}")
        logger.info("Continuing with existing index (if any)...")

    # 2. Get Langfuse client
    langfuse = get_langfuse_client()

    # 3. Load dataset
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = dataset.items

    logger.info("\n" + "=" * 60)
    logger.info(f"Loaded dataset '{DATASET_NAME}' from Langfuse")
    logger.info(f"Total items: {len(items)}")
    logger.info("=" * 60)

    results: List[Dict[str, Any]] = []

    # 4. Run evaluation for each dataset item
    for idx, item in enumerate(items, start=1):
        question = item.input.get("question")
        expected_output = item.expected_output

        logger.info(f"\n--- Dataset item {idx}/{len(items)} ---")
        logger.info(f"Q: {question}")
        logger.info(f"Expected: {expected_output}")

        result = await query_with_evaluation(
            langfuse=langfuse,
            question=question,
            expected_output=expected_output,
            similarity_top_k=2,
        )
        results.append(result)

        # Small delay to avoid rate limits
        await asyncio.sleep(1)

    # 5. Aggregate metrics
    if results:
        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)

        # Calculate SLO compliance
        faith_slo = 0.8
        relevance_slo = 0.75
        
        faith_compliant = sum(1 for r in results if r["faithfulness"] >= faith_slo)
        relevance_compliant = sum(1 for r in results if r["answer_relevance"] >= relevance_slo)
        
        faith_compliance_rate = (faith_compliant / len(results)) * 100
        relevance_compliance_rate = (relevance_compliant / len(results)) * 100

        # Find lowest scoring questions
        lowest_faithfulness = min(results, key=lambda x: x["faithfulness"])
        lowest_relevance = min(results, key=lambda x: x["answer_relevance"])

        logger.info("\n" + "=" * 60)
        logger.info("Dataset Evaluation Complete")
        logger.info("=" * 60)
        logger.info(f"Total queries: {len(results)}")
        logger.info(f"\nAverage Scores:")
        logger.info(f"  Average Faithfulness:     {avg_faithfulness:.2f}")
        logger.info(f"  Average Answer Relevance: {avg_relevance:.2f}")
        logger.info(f"\nSLO Compliance:")
        logger.info(f"  Faithfulness SLO (≥{faith_slo}): {faith_compliant}/{len(results)} ({faith_compliance_rate:.1f}%)")
        logger.info(f"  Answer Relevance SLO (≥{relevance_slo}): {relevance_compliant}/{len(results)} ({relevance_compliance_rate:.1f}%)")
        logger.info(f"\nLowest Scoring Questions:")
        logger.info(f"  Faithfulness: {lowest_faithfulness['question'][:60]}... (Score: {lowest_faithfulness['faithfulness']:.2f})")
        logger.info(f"  Answer Relevance: {lowest_relevance['question'][:60]}... (Score: {lowest_relevance['answer_relevance']:.2f})")
        logger.info(f"\nCheck detailed traces & scores in Langfuse Dashboard (Traces + Scores).")
        logger.info("=" * 60)
        
        return results


if __name__ == "__main__":
    asyncio.run(run_dataset_evaluation())

# MedLearn Assistant – AI-Powered Medical Education Companion

## 📌 Problem Statement
Medical and healthcare students face an overwhelming volume of complex information spread across dense textbooks, research papers, and clinical guidelines. Learning anatomy, physiology, and pathology requires interpreting intricate diagrams, tables, and technical terminology—often under intense academic and clinical time pressure.

In addition, students must quickly access structured clinical data such as drug interactions, laboratory reference ranges, and contraindications. Existing tools fail to seamlessly combine conceptual understanding with precise clinical facts.

---

## 🌍 Context
MedLearn Assistant is designed to bridge this gap by providing an AI-powered learning companion that intelligently retrieves and synthesizes information from **medical textbooks** and **structured clinical databases** using natural language queries.

---

## 🚧 Key Real-World Challenges
- Medical textbooks contain **multi-modal content** (text, diagrams, tables) that is difficult to search efficiently.
- Students waste significant time manually navigating large textbooks to find specific information.
- Conceptual explanations (from textbooks) and factual lookups (from databases) are handled by separate tools.
- Current systems lack **intelligent query routing** to select or combine the right data source.

---

## 🎯 Project Goal
Build a **production-grade AI-powered MedLearn Assistant** that:
- Instantly retrieves and explains complex medical information
- Intelligently routes queries to unstructured content (RAG), structured databases (SQL), or a hybrid of both
- Provides accurate, contextualized answers with proper citations and safety guardrails

---

## 🧠 Problem Description
This project focuses on building a **Production-Grade RAG System for Medical Education** that:

- Ingests complex medical textbooks (PDFs with text, diagrams, tables)
- Integrates structured clinical databases (drug interactions, lab ranges, contraindications)
- Routes queries intelligently:
  - **RAG** → Conceptual explanations
  - **SQL** → Deterministic factual lookups
  - **Hybrid** → Combined answers
- Supports natural language queries such as:
  - *“Explain the difference between rapidly and slowly adapting receptors”*
  - *“What are the major drug interactions with warfarin?”*
- Handles multi-modal medical content
- Integrates external medical APIs (e.g., PubMed)
- Implements medical safety guardrails
- Evaluates performance against defined SLOs
- Supports human expert handoff for complex queries

## Hybrid query agent

`POST /query` is served by an OpenAI-compatible agent (Azure OpenAI is supported) with these tools:

- `clinical_sql` for the allow-listed clinical tables in `medical_database_schema.sql`
- `semantic_search`, `bm25_search`, and `hybrid_document_search` for the pgvector document index
- Hugging Face MCP tools, when `HF_TOKEN` and `HF_MCP_URL` are configured

The response includes `tools_used`, `tool_errors`, and `routing`. Individual tool setup or execution errors are returned as metadata while the agent tries the remaining capabilities; an unavailable SQL database or missing document index does not take down the API.

---

## ⚙️ Functional Requirements
Your system must support:

### 📄 Document Processing
- Ingest medical textbooks (PDFs) with complex formatting

### 🖼️ Multi-Modal Understanding
- Extract and retrieve insights from text, diagrams, and tables

### 🧭 Intelligent Query Routing
- Decide between RAG, SQL, or Hybrid retrieval dynamically

### 💬 Natural Language Queries
- Conversational medical questions across anatomy, physiology, and clinical data

### 🗄️ Structured Data Integration
- SQL-based lookup for deterministic clinical facts

### 🔄 Hybrid Retrieval
- Combine conceptual context with structured facts

### 🔌 External Integrations
- Medical APIs via Model Context Protocol (MCP)

### 🛡️ Accuracy & Safety
- Guardrails to prevent medical misinformation

### 📊 Performance Evaluation
- Accuracy, latency, and retrieval quality measured against SLOs

### 👨‍⚕️ Human Handoff
- Escalation workflow with full context transfer

---

## 🧪 Technical Details

### 🧑‍💻 Programming Language
- **Python**

### 🏗️ Core Framework
- **LlamaIndex**

### 🧰 Libraries & Tools

| Tool / Library | Purpose |
|----------------|--------|
| llamaindex | Build advanced multi-modal RAG pipelines |
| pgvector | Vector storage and similarity search |
| sqlalchemy | SQL database integration |
| fastapi | Backend API |
| uvicorn | FastAPI server |
| guardrails-ai | Medical accuracy guardrails |
| langfuse / ragas | RAG evaluation |
| anthropic / openai | LLM APIs |
| dotenv | Environment variable management |
| pymupdf / pypdf | PDF extraction |

---

## 🔐 Environment Variables

| Variable | Purpose |
|--------|--------|
| ANTHROPIC_API_KEY | Claude API authentication |
| OPENAI_API_KEY | OpenAI API authentication (optional) |
| DATABASE_URL | PostgreSQL connection with pgvector |
| MCP_SERVER_URL | MCP endpoint |

---

## 🏗️ Infrastructure Requirements
- PostgreSQL with **pgvector** extension
- Structured medical SQL database
- Vector embedding model (OpenAI / Cohere)
- Document storage (local or cloud)

---

## 📚 Sample Dataset
- Somatosensory System textbook chapter (Wikibooks)
- Additional anatomy & physiology textbooks
- Structured clinical database (drug interactions, lab ranges)
- Optional: PubMed articles

---

## 📦 Project Deliverables

### 1️⃣ Functional RAG System
- Document ingestion pipeline
- Multi-modal content processing
- Natural language query interface
- Intelligent query routing
- SQL integration
- External API (MCP) integration

### 2️⃣ Safety & Quality
- Medical guardrails
- Response validation
- Source attribution

### 3️⃣ Performance Evaluation
- SLO-based metrics
- Query routing effectiveness
- Benchmark report

### 4️⃣ Human Handoff
- Context transfer mechanism
- Expert escalation workflow

### 5️⃣ Documentation
- Architecture diagram
- Query routing logic
- API documentation
- Deployment guide

---

## 🧪 Evaluation Criteria
The system will be evaluated on:

- Retrieval accuracy
- Query routing intelligence
- Multi-modal processing capability
- Response quality and medical correctness
- System performance
- Guardrail effectiveness
- Code quality
- Documentation clarity

---

## 🚀 Getting Started

1. Set up PostgreSQL with pgvector
2. Create structured medical database
3. Ingest textbook content
4. Build RAG pipeline with LlamaIndex
5. Implement query routing
6. Implement RAG and SQL engines
7. Add multi-modal processing
8. Integrate MCP APIs
9. Implement guardrails and evaluation
10. Test with sample queries
11. Document architecture and findings

> **Note:** Focus on building a production-grade system demonstrating intelligent data source selection beyond basic document Q&A.

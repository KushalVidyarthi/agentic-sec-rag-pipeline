# Agentic RAG Pipeline for Financial Filings (SEC 10-K)

A production-style, multi-phase Retrieval-Augmented Generation (RAG) system that lets you interrogate SEC 10-K annual filings using natural language. The pipeline combines structured table extraction, dense vector search, intent-based semantic routing, and a Google Gemini-backed LLM agent to answer both narrative and numerical questions with guardrails against hallucination.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│     Semantic Router             │  cosine-similarity intent dispatch
│     (semantic_router.py)        │  (all-MiniLM-L6-v2 embeddings)
└────────────┬────────────────────┘
             │
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
TEXT_INTENT          DATA_INTENT
    │                    │
    ▼                    ▼
ChromaDB RAG      Pandas LLM Agent
(chroma_db/)      (Gemini 1.5 Flash)
narrative text    structured tables
    │                    │
    └────────┬───────────┘
             │
    ┌────────▼────────────┐
    │  Guardrails &       │
    │  Agentic Fallback   │  relevance threshold · agent-error fallback
    └─────────────────────┘
```

---

## Project Structure

| File | Phase | Role |
|------|-------|------|
| [`extract_financials.py`](extract_financials.py) | 1 – Ingestion | Extracts the *Consolidated Statements of Operations* from a PDF using `pdfplumber`'s coordinate-aware table parser, then structures it into a Pandas DataFrame. |
| [`build_vector_store.py`](build_vector_store.py) | 2 – Vector Engine | Reads the full 10-K PDF, chunks it with `RecursiveCharacterTextSplitter` (1 000 chars / 200 overlap), embeds chunks locally with `all-MiniLM-L6-v2`, and persists them to ChromaDB. |
| [`semantic_router.py`](semantic_router.py) | 3 – Router | Embeds the incoming query and measures cosine similarity against two pre-defined intent anchors (`TEXT_INTENT`, `DATA_INTENT`) to dispatch to the correct retrieval path. |
| [`agent_pipeline.py`](agent_pipeline.py) | 4 – Agent | Unified interactive CLI. Orchestrates all prior phases: runs the semantic router, executes either the RAG text path or the Gemini Pandas agent, applies relevance guardrails, and falls back between paths on failure. |
| [`validate_tiers.py`](validate_tiers.py) | Testing | Smoke-tests the tiered query logic end-to-end (Tier 1 direct lookups, Tier 2 LLM escalation). |

---

## Query Lifecycle

```
User query
    └─► Semantic Router (cosine similarity)
            ├─► TEXT_INTENT ──► Chroma vector search
            │                       ├─► score ≥ 0.35 → return top-K chunks
            │                       └─► score < 0.35 → GUARDRAIL: "out of scope"
            └─► DATA_INTENT ──► Pandas LLM Agent (Gemini)
                                    ├─► agent succeeds → return answer
                                    └─► agent fails    → FALLBACK to text path
```

---

## Tech Stack

| Layer | Library |
|-------|---------|
| PDF parsing | `pdfplumber` |
| Text splitting | `langchain-text-splitters` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | `ChromaDB` (`langchain-community`) |
| LLM | Google Gemini via `langchain-google-genai` |
| DataFrame agent | `langchain-experimental` |
| Orchestration | `LangGraph`, `LangChain` |
| Data wrangling | `pandas`, `numpy` |

---

## Quick Start

### 1. Clone & create a virtual environment

```bash
git clone <repo-url>
cd "Agentic RAG Pipeline for Financial Filings"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Copy the template and fill in your keys
cp .env.example .env
```

Open `.env` and set at minimum:

```env
GOOGLE_API_KEY=your-google-api-key-here
```

Get a free Gemini API key at <https://aistudio.google.com/app/apikey>.

### 4. Add your 10-K PDF

Place your SEC 10-K PDF in the project root and update `PDF_PATH` in `build_vector_store.py` if it differs from the default (`apple_10k.pdf`).

### 5. Build the vector store (one-time)

```bash
python build_vector_store.py
```

This reads the PDF, chunks it, embeds every chunk locally (no API key required), and writes the ChromaDB collection to `./chroma_db/`.

### 6. Launch the interactive pipeline

```bash
python agent_pipeline.py
```

Type any natural-language question about the filing. The semantic router will dispatch it to the appropriate retrieval path automatically.

**Example queries:**
- *"What was Apple's net income in 2024?"* → DATA_INTENT → Pandas agent
- *"Describe the key risk factors for Apple's supply chain."* → TEXT_INTENT → ChromaDB RAG
- *"What were total net sales compared to last year?"* → DATA_INTENT → Pandas agent

### 7. (Optional) Validate the tiered query logic

```bash
python validate_tiers.py
```

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | **Yes** | Gemini API key for the LLM Pandas agent (Phase 4). |
| `OPENAI_API_KEY` | No | Optional; only needed if switching LLM backend to OpenAI. |
| `POSTGRES_URI` | No | PostgreSQL connection string for external structured storage. |
| `CHROMA_PERSIST_DIR` | No | Override the default ChromaDB directory (`./chroma_db`). |

---

## Key Design Decisions

- **Local embeddings** — `all-MiniLM-L6-v2` runs entirely on-device; no API quota is consumed for embedding.
- **Relevance guardrail** — Text-path answers are withheld when the best chunk scores below 0.35 cosine similarity, preventing hallucinated answers to out-of-scope questions.
- **Agentic fallback** — If the Pandas agent fails on a DATA_INTENT query, the pipeline automatically retries through the ChromaDB text path rather than returning an error.
- **pdfplumber over text loaders** — Coordinate-aware table parsing preserves the multi-column grid structure of financial statements; text loaders flatten columns and corrupt numerical data.

---

## License

MIT

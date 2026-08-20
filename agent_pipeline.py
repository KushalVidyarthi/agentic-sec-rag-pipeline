"""
agent_pipeline.py
-----------------
Phase 4 -- Unified Agentic RAG Pipeline: End-to-End Query Engine

Ties together all prior phases into a single interactive system:

  Phase 1  extract_financials.py  --> Structured DataFrame (tabular financial data)
  Phase 2  ./chroma_db            --> Persistent vector store (narrative text chunks)
  Phase 3  Semantic Router        --> Cosine-similarity intent dispatcher
  Phase 4  [THIS FILE]            --> Execution agents + guardrails + interactive CLI

Query lifecycle:
  User query
      │
      ▼
  Semantic Router (cosine similarity vs. TEXT / DATA intent anchors)
      │
      ├── TEXT_INTENT --> execute_text_query()
      │                       │
      │                       ├── relevance_score >= 0.35  → return top-K chunks
      │                       └── relevance_score <  0.35  → GUARDRAIL TRIGGERED
      │
      └── DATA_INTENT --> execute_tabular_query()  [LLM-backed Pandas agent]
                              │
                              ├── agent succeeds  → return LLM answer
                              └── agent fails     → [AGENTIC FALLBACK]
                                                       │
                                                       └── execute_text_query()
                                                           (same guardrail rules apply)
"""

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load .env from the project directory — populates GOOGLE_API_KEY before
# any LangChain/Google SDK imports that read environment variables at import time.
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.vectorstores import Chroma

# Pull the two reusable functions from Phase 1 without re-running its __main__
from extract_financials import find_target_page, extract_financial_table


# ─────────────────────────────────────────────
# PIPELINE CONFIGURATION
# ─────────────────────────────────────────────

PDF_PATH: str            = "apple_10k.pdf"
TARGET_PHRASE: str       = "CONSOLIDATED STATEMENTS OF OPERATIONS"
EMBEDDING_MODEL: str     = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR: str  = "./chroma_db"
CHROMA_COLLECTION: str   = "sec_10k_text"

# Routing anchors — same wording as Phase 3 for consistent intent space.
TEXT_INTENT: str = (
    "Questions about company risks, future outlook, leadership, or textual summaries."
)
DATA_INTENT: str = (
    "Questions about revenue, balance sheets, calculations, or comparing financial numbers."
)

# Guardrail threshold: retrieved chunks with relevance score below this value
# are considered insufficient context and trigger the safe fallback message.
# Chroma's relevance scores are normalised cosine similarities (0.0–1.0).
RELEVANCE_THRESHOLD: float = 0.35

# Number of RAG chunks to surface for text queries.
TOP_K_TEXT: int = 3

# ── Pandas DataFrame Agent (LLM-backed) ──────────────────────────────────────
# Google Gemini is used as the reasoning engine for the analytical agent.
# Set GOOGLE_API_KEY in your environment or .env before running.
GOOGLE_API_KEY: str    = os.getenv("GOOGLE_API_KEY", "")
PANDAS_AGENT_MODEL: str = "gemini-3.6-flash"
# Temperature=0 for deterministic, reproducible pandas code generation.
PANDAS_AGENT_TEMP: float = 0.0


# ─────────────────────────────────────────────
# STARTUP: LOAD ALL SHARED RESOURCES ONCE
# ─────────────────────────────────────────────

def _load_resources() -> tuple[pd.DataFrame, Chroma, np.ndarray, np.ndarray, HuggingFaceEmbeddings]:
    """
    Initialise all pipeline components at startup.

    Loading order matters:
      1. Embedding model  (needed by both Chroma loader and anchor embedder)
      2. Chroma           (reads from disk — instant, no embedding work)
      3. Financial DataFrame  (pdfplumber scan + table extraction)
      4. Intent anchors   (two embed_query calls)

    Returns a tuple of (df, vector_store, text_anchor, data_anchor, embeddings)
    so callers can hold references to each component independently.
    """
    print("[INIT] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[INIT] Loading Chroma vector store from disk...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
        )

    print(f"[INIT] Extracting financial DataFrame from '{PDF_PATH}'...")
    page_num = find_target_page(PDF_PATH, TARGET_PHRASE)
    df = extract_financial_table(PDF_PATH, page_num)
    if df is None:
        raise RuntimeError(
            "Phase 1 table extraction returned None — check extract_financials.py config."
        )

    print("[INIT] Embedding intent anchors...")
    text_anchor = np.array(embeddings.embed_query(TEXT_INTENT))
    data_anchor = np.array(embeddings.embed_query(DATA_INTENT))

    print("[INIT] All resources loaded. Pipeline ready.\n")
    return df, vector_store, text_anchor, data_anchor, embeddings


# ─────────────────────────────────────────────
# NORMALISE FINANCIAL DATAFRAME
# ─────────────────────────────────────────────

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw DataFrame from extract_financial_table() uses the first data row
    as column headers (a pdfplumber text-strategy artifact). Standardise it
    into a clean structure:

        item | val_2025 | val_2024 | val_2023

    The pdfplumber text strategy produces alternating value/None columns for
    dollar-sign artifacts. We drop the None-dominated columns and keep only
    the item name column and the three numeric value columns.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with columns: ['item', '2025', '2024', '2023']
        and one row per financial line item.
    """
    cols = list(df.columns)

    # The extraction leaves the "Products" row in the header because it was the
    # first data row. Reconstruct it as a proper data row.
    header_values = [str(c) for c in cols]

    # Column layout produced by pdfplumber text strategy (7-col case):
    #
    #   col name  : 'Products'  '$'        '307,003'  '$_1'      '294,866'  '$_2'      '298,085'
    #   col index :  0           1          2          3          4          5          6
    #
    # HETEROGENEOUS layout — two patterns exist across rows:
    #
    #   Pattern A (most rows):  item | VALUE  | None  | VALUE  | None  | VALUE  | None
    #     e.g. Services:        Serv | 109158 | None  | 96169  | None  | 85200  | None
    #
    #   Pattern B (Net income, EPS):  item | $  | VALUE  | $  | VALUE  | $  | VALUE
    #     e.g. Net income:            Net  | $  | 112010 | $  | 93736  | $  | 96995
    #
    # Solution: for each year, merge the odd/even column pair — take the value
    # from the odd column (1,3,5) if it's a parseable number, else use the even
    # column (2,4,6). This handles both patterns in a single pass.
    n = len(cols)
    if n < 4:
        return df

    item_col = cols[0]

    def _is_dollar_marker(val) -> bool:
        """True if the cell is a bare currency marker, not a real number."""
        return str(val).strip() in ("$", "None", "") or val is None

    def _merge_pair(odd_idx: int, even_idx: int) -> list:
        """
        For each row, take the value at odd_idx (positional) when it is a real
        number; otherwise fall back to even_idx. Uses iloc to access by position,
        not by column name (which would raise KeyError for integer keys).
        """
        col_a = df.iloc[:, odd_idx]
        col_b = df.iloc[:, even_idx] if even_idx < n else [None] * len(df)
        return [b if _is_dollar_marker(a) else a for a, b in zip(col_a, col_b)]

    if n >= 7:
        merged_2025 = _merge_pair(1, 2)
        merged_2024 = _merge_pair(3, 4)
        merged_2023 = _merge_pair(5, 6)
    elif n >= 5:
        merged_2025 = _merge_pair(1, 2)
        merged_2024 = _merge_pair(3, 4)
        merged_2023 = list(df.iloc[:, 5] if n > 5 else [None] * len(df))
    else:
        merged_2025 = list(df.iloc[:, 1])
        merged_2024 = list(df.iloc[:, 2] if n > 2 else [None] * len(df))
        merged_2023 = list(df.iloc[:, 3] if n > 3 else [None] * len(df))

    # Reconstruct the Products header row using the same merge logic on
    # the column names (header_values), which carry the Products values.
    col1_is_currency = _is_dollar_marker(cols[1])
    if col1_is_currency and n >= 7:
        h2025, h2024, h2023 = header_values[2], header_values[4], header_values[6]
    else:
        h2025 = header_values[1] if n > 1 else None
        h2024 = header_values[3] if n > 3 else None
        h2023 = header_values[5] if n > 5 else None

    reconstructed_first_row = {
        "item" : header_values[0],
        "2025" : h2025,
        "2024" : h2024,
        "2023" : h2023,
    }

    slim = pd.DataFrame({
        "item" : list(df[item_col]),
        "2025" : merged_2025,
        "2024" : merged_2024,
        "2023" : merged_2023,
    })

    # Prepend the recovered header row.
    first_row_df = pd.DataFrame([reconstructed_first_row])
    slim = pd.concat([first_row_df, slim], ignore_index=True)

    # Drop rows where the item name is None or whitespace (separator rows).
    slim = slim[slim["item"].notna() & (slim["item"].str.strip() != "")]
    slim.reset_index(drop=True, inplace=True)

    return slim


def _parse_value(val_str) -> float | None:
    """
    Convert a raw financial string like "307,003" or "(321)" to a float.
    Parentheses denote negative values per GAAP accounting convention.
    Returns None if the string cannot be parsed.
    """
    if val_str is None or str(val_str).strip() in ("", "None"):
        return None
    s = str(val_str).strip().replace(",", "")
    # GAAP parenthetical negatives: (321) → -321
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


# ─────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity. Degrades to dot product for unit vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ─────────────────────────────────────────────
# SEMANTIC ROUTER
# ─────────────────────────────────────────────

def _route(query: str, embeddings: HuggingFaceEmbeddings,
           text_anchor: np.ndarray, data_anchor: np.ndarray) -> tuple[str, float, float, float]:
    """
    Embed *query* and return the winning intent route plus both similarity
    scores and the decision margin.

    Returns
    -------
    tuple[str, float, float, float]
        (winning_intent, sim_text, sim_data, margin)
        winning_intent is either "TEXT" or "DATA".
    """
    q_vec = np.array(embeddings.embed_query(query))
    sim_text = _cosine_sim(q_vec, text_anchor)
    sim_data = _cosine_sim(q_vec, data_anchor)
    winner = "TEXT" if sim_text >= sim_data else "DATA"
    margin = abs(sim_text - sim_data)
    return winner, sim_text, sim_data, margin


# ─────────────────────────────────────────────
# EXECUTION AGENT A: TEXT RETRIEVAL + GUARDRAILS
# ─────────────────────────────────────────────

def execute_text_query(query: str, vector_store: Chroma) -> dict:
    """
    Retrieve the most relevant narrative chunks from the Chroma vector store.

    Guardrail: if the top chunk's relevance score (normalised cosine similarity)
    falls below RELEVANCE_THRESHOLD, the pipeline refuses to answer rather than
    risk surfacing a hallucinated or misattributed response.

    Parameters
    ----------
    query        : str   -- The user's natural-language question.
    vector_store : Chroma

    Returns
    -------
    dict with keys:
        "answer"    : str  -- The formatted answer or guardrail message.
        "guardrail" : bool -- True if the guardrail was triggered.
        "scores"    : list[float] -- Relevance scores for each retrieved chunk.
        "chunks"    : list[str]   -- The raw chunk text for each result.
    """
    # similarity_search_with_relevance_scores returns List[(Document, float)]
    # where the float is a normalised similarity score in [0, 1].
    results_with_scores = vector_store.similarity_search_with_relevance_scores(
        query, k=TOP_K_TEXT
    )

    if not results_with_scores:
        return {
            "answer"    : "GUARDRAIL TRIGGERED: No documents found in vector store.",
            "guardrail" : True,
            "scores"    : [],
            "chunks"    : [],
        }

    top_score = results_with_scores[0][1]

    # ── Guardrail check ───────────────────────────────────────────────────────
    # A low top-score means even the closest chunk in embedding space is not
    # semantically near the query — the vector store simply does not contain
    # reliable context to answer this question.
    if top_score < RELEVANCE_THRESHOLD:
        return {
            "answer"    : (
                "GUARDRAIL TRIGGERED: Insufficient context found in SEC 10-K to reliably "
                "answer this query without hallucination."
            ),
            "guardrail" : True,
            "scores"    : [s for _, s in results_with_scores],
            "chunks"    : [d.page_content for d, _ in results_with_scores],
        }

    # ── Format retrieved context ──────────────────────────────────────────────
    answer_parts = []
    for rank, (doc, score) in enumerate(results_with_scores, 1):
        preview = doc.page_content.replace("\n", " ").strip()
        answer_parts.append(
            f"  [Source {rank}]  relevance={score:.4f}\n"
            f"  {preview[:500]}{'...' if len(preview) > 500 else ''}"
        )

    return {
        "answer"    : "\n\n".join(answer_parts),
        "guardrail" : False,
        "scores"    : [s for _, s in results_with_scores],
        "chunks"    : [d.page_content for d, _ in results_with_scores],
    }


# ─────────────────────────────────────────────
# EXECUTION AGENT B: LLM-BACKED PANDAS AGENT
# ─────────────────────────────────────────────


def execute_tabular_query(query: str, df: pd.DataFrame) -> dict:
    """
    Answer an analytical question using a 3-tier hybrid execution strategy.

    Tier 1  — Fast Path (local, zero API calls)
        Iterate df rows and check whether the lowercase row label appears in
        the user's query. On a match, parse and format the numeric values
        immediately. Handles the vast majority of simple lookups in < 1 ms.

    Tier 2  — LLM Escalation (Gemini Pandas Agent)
        Triggered only when Tier 1 finds no match — signals a complex or
        ambiguous query (e.g. "Services gross margin" requiring arithmetic
        across two separate rows). Invokes create_pandas_dataframe_agent with
        the full DataFrame and a schema-aware context prompt.

    Tier 3  — Error Catch → Agentic Fallback
        Any exception from the LLM layer (quota exceeded, bad codegen, network
        timeout, missing API key) is caught here. Returns matched="none" with
        an empty raw_row, which run_query's fallback loop detects and
        automatically reroutes to execute_text_query (Chroma RAG).

    Parameters
    ----------
    query : str          -- The user's natural-language analytical question.
    df    : pd.DataFrame -- Normalised financial DataFrame from _normalise_df().

    Returns
    -------
    dict with keys:
        "answer"  : str  -- Formatted answer string.
        "matched" : str  -- Row label, "llm_agent", or "none" (fallback sentinel).
        "raw_row" : dict -- Parsed {year: float} values (Tier 1) or {} (Tier 2/3).
    """

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1 — FAST LOCAL FUZZY MATCH
    # ══════════════════════════════════════════════════════════════════════════
    # Scan every DataFrame row; check if the normalised row label is a
    # substring of the normalised query. Prefer longer matches to avoid
    # collisions (e.g. "total net sales" should beat "sales" alone).

    q_lower = query.lower().strip()

    best_match_row   = None
    best_match_label = ""
    best_match_len   = 0

    for _, row in df.iterrows():
        raw_label = str(row.get("item", "") or "").strip()
        if not raw_label:
            continue
        label_lower = raw_label.lower()

        # Require the label to appear as a substring of the query.
        if label_lower in q_lower and len(label_lower) > best_match_len:
            # Validate: at least one year column must contain parseable data.
            v25 = _parse_value(row.get("2025"))
            v24 = _parse_value(row.get("2024"))
            v23 = _parse_value(row.get("2023"))
            if any(v is not None for v in [v25, v24, v23]):
                best_match_row   = row
                best_match_label = raw_label
                best_match_len   = len(label_lower)
                # Cache parsed values alongside the row so we don't re-parse.
                _best_vals = (v25, v24, v23)

    if best_match_row is not None:
        v2025, v2024, v2023 = _best_vals
        raw_row = {"2025": v2025, "2024": v2024, "2023": v2023}

        def _fmt(v: float | None) -> str:
            return f"${v:,.0f}M" if v is not None else "N/A"

        is_comparison = any(kw in q_lower for kw in
                            ["increase", "decrease", "change", "grew", "compare",
                             "higher", "lower", "more", "less", "vs", "versus", "differ"])

        if is_comparison and v2025 is not None and v2024 is not None:
            delta     = v2025 - v2024
            pct       = (delta / abs(v2024) * 100) if v2024 != 0 else float("inf")
            direction = "increased" if delta >= 0 else "decreased"
            answer = (
                f"[TIER-1 FAST PATH]\n"
                f"  Line item  : {best_match_label}\n"
                f"  FY2025     : {_fmt(v2025)}\n"
                f"  FY2024     : {_fmt(v2024)}\n"
                f"  FY2023     : {_fmt(v2023)}\n"
                f"  YoY Change : {direction} by {_fmt(abs(delta))} ({pct:+.1f}%)\n"
                f"  (Source: Apple 10-K, Consolidated Statements of Operations)"
            )
        else:
            answer = (
                f"[TIER-1 FAST PATH]\n"
                f"  Line item  : {best_match_label}\n"
                f"  FY2025     : {_fmt(v2025)}\n"
                f"  FY2024     : {_fmt(v2024)}\n"
                f"  FY2023     : {_fmt(v2023)}\n"
                f"  (Source: Apple 10-K, Consolidated Statements of Operations)"
            )

        return {"answer": answer, "matched": best_match_label, "raw_row": raw_row}

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2 — LLM ESCALATION (Gemini Pandas Agent)
    # ══════════════════════════════════════════════════════════════════════════
    # Tier 1 found no single-row match — the query likely requires cross-row
    # arithmetic (e.g. gross margin = revenue minus cost across duplicate row
    # names) or phrasing the static matcher can't resolve. Hand off to Gemini.

    print("  [ESCALATION] Complex query detected. Escalating to LLM Pandas Agent...")

    try:
        llm = ChatGoogleGenerativeAI(
            model=PANDAS_AGENT_MODEL,
            temperature=PANDAS_AGENT_TEMP,
            google_api_key=GOOGLE_API_KEY,
        )

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            verbose=False,
        )

        context_prompt = (
            f"{query}\n\n"
            "The DataFrame `df` has columns: 'item' (line item name), "
            "'2025' (FY2025 value), '2024' (FY2024 value), '2023' (FY2023 value). "
            "Values are strings with commas as thousands separators "
            "(e.g. '416,161') and may contain None for non-numeric rows. "
            "Strip commas and cast to float before any arithmetic. "
            "Return a concise, human-readable answer with the specific numbers."
        )

        response   = agent.invoke({"input": context_prompt})
        raw_output = response.get("output", "") if isinstance(response, dict) else str(response)
        raw_output = str(raw_output).strip()

        if not raw_output:
            raise ValueError("Pandas agent returned an empty response.")

        answer = (
            f"[TIER-2 LLM PANDAS AGENT]\n"
            f"  {raw_output}\n"
            f"  (Source: Apple 10-K, Consolidated Statements of Operations)"
        )
        return {"answer": answer, "matched": "llm_agent", "raw_row": {}}

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 3 — ERROR CATCH → AGENTIC FALLBACK SENTINEL
    # ══════════════════════════════════════════════════════════════════════════
    # Quota errors, network timeouts, bad codegen, missing API key — all land
    # here. Returning matched="none" with an empty raw_row triggers run_query's
    # fallback loop which reroutes the query to execute_text_query (Chroma RAG).

    except Exception as exc:
        err_type = type(exc).__name__
        print(f"  [PANDAS AGENT ERROR] {err_type}: {exc}")
        return {
            "answer"  : f"LLM agent failed ({err_type}). Triggering Chroma fallback.",
            "matched" : "none",
            "raw_row" : {},
        }


# ─────────────────────────────────────────────
# UNIFIED CONTROLLER
# ─────────────────────────────────────────────

def run_query(
    user_query: str,
    df: pd.DataFrame,
    vector_store: Chroma,
    embeddings: HuggingFaceEmbeddings,
    text_anchor: np.ndarray,
    data_anchor: np.ndarray,
) -> None:
    """
    Route *user_query* through the semantic router and dispatch to the
    appropriate execution agent. Prints a formatted response block including
    routing metadata and wall-clock latency.

    Parameters
    ----------
    user_query   : str
    df           : pd.DataFrame        -- Normalised financial table (Phase 1).
    vector_store : Chroma              -- Loaded from disk (Phase 2).
    embeddings   : HuggingFaceEmbeddings
    text_anchor  : np.ndarray
    data_anchor  : np.ndarray
    """
    print("\n" + "=" * 70)
    print(f"  QUERY : {user_query}")
    print("=" * 70)

    t_start = time.perf_counter()

    # ── Route ─────────────────────────────────────────────────────────────────
    winner, sim_text, sim_data, margin = _route(
        user_query, embeddings, text_anchor, data_anchor
    )

    t_routed = time.perf_counter()

    print(f"  [ROUTER] TEXT={sim_text:.4f}  DATA={sim_data:.4f}  "
          f"Winner={winner}_INTENT  margin={margin:.4f}")

    # ── Global low-confidence guardrail ───────────────────────────────────────
    # If BOTH anchors score below 0.10 the query is almost certainly out-of-domain
    # (e.g. general knowledge questions). Refuse immediately rather than dispatching
    # to an agent that has no chance of producing a grounded answer.
    LOW_CONFIDENCE_FLOOR = 0.10
    if max(sim_text, sim_data) < LOW_CONFIDENCE_FLOOR:
        t_end = time.perf_counter()
        print(
            "  GUARDRAIL TRIGGERED: Query appears out-of-domain for SEC 10-K filings.\n"
            "  Both intent anchors scored below the confidence floor "
            f"({max(sim_text, sim_data):.4f} < {LOW_CONFIDENCE_FLOOR}).\n"
            "  Please ask a question about Apple's financial statements or disclosures."
        )
        print(f"\n  -- Routing branch : GLOBAL_GUARDRAIL")
        print(f"  -- Total latency  : {(t_end - t_start)*1000:.1f} ms")
        print("=" * 70)
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    if winner == "TEXT":
        print(f"  [DISPATCH] --> Text Retrieval Agent (guardrail threshold={RELEVANCE_THRESHOLD})\n")
        result = execute_text_query(user_query, vector_store)
        branch = "TEXT_RAG"
        if result["guardrail"]:
            top_score = result["scores"][0] if result["scores"] else 0.0
            print(f"  [GUARDRAIL] Top relevance score: {top_score:.4f} < {RELEVANCE_THRESHOLD}")
    else:
        print(f"  [DISPATCH] --> Analytical Execution Agent\n")
        result = execute_tabular_query(user_query, df)
        branch = "DATA_TABULAR"

        # ── Agentic Fallback Loop ─────────────────────────────────────────────
        # A clean miss from the tabular agent (matched="none", empty raw_row)
        # means the line item either doesn't exist in the structured table or
        # its label is too corrupted to match even after fuzzy normalisation.
        # Rather than surfacing an unhelpful error, we automatically re-route
        # to the text retrieval agent — it may hold the answer in narrative
        # sections (MD&A, Notes to Financial Statements, etc.).
        if result["matched"] == "none" and not result["raw_row"]:
            print("  [AGENTIC FALLBACK] Tabular match failed. "
                  "Rerouting to Text Retrieval Agent...\n")
            result = execute_text_query(user_query, vector_store)
            branch = "DATA_TABULAR->TEXT_RAG_FALLBACK"
            if result["guardrail"]:
                top_score = result["scores"][0] if result["scores"] else 0.0
                print(f"  [GUARDRAIL] Fallback relevance score: "
                      f"{top_score:.4f} < {RELEVANCE_THRESHOLD}")

    t_end = time.perf_counter()

    # ── Output ────────────────────────────────────────────────────────────────
    print(result["answer"])
    print()
    print(f"  -- Routing branch : {branch}")
    print(f"  -- Router latency : {(t_routed - t_start)*1000:.1f} ms")
    print(f"  -- Total latency  : {(t_end   - t_start)*1000:.1f} ms")
    print("=" * 70)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Startup ───────────────────────────────────────────────────────────────
    df_raw, vector_store, text_anchor, data_anchor, embeddings = _load_resources()

    # Normalise the DataFrame once at startup — all queries share it.
    df = _normalise_df(df_raw)

    # ── Built-in test cases ───────────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("  BUILT-IN TEST CASES")
    print("#" * 70)

    run_query(
        "What are the major supply chain risks facing the company?",
        df, vector_store, embeddings, text_anchor, data_anchor,
    )

    run_query(
        "Did the gross margin for the services sector increase in 2025?",
        df, vector_store, embeddings, text_anchor, data_anchor,
    )

    run_query(
        "What was the total net sales in 2025 compared to 2024?",
        df, vector_store, embeddings, text_anchor, data_anchor,
    )

    run_query(
        "What is the capital of France?",   # out-of-domain → global guardrail test
        df, vector_store, embeddings, text_anchor, data_anchor,
    )

    # Fuzzy matching test: query uses a clean label but the PDF row may contain
    # trailing footnote markers like "Net income (1) " — the three-tier fuzzy
    # matcher should resolve this without human intervention.
    run_query(
        "What was Apple's net income in 2025?",
        df, vector_store, embeddings, text_anchor, data_anchor,
    )

    # Agentic fallback test: "capital expenditure" is a DATA-intent query but
    # is NOT a line item in the Consolidated Statements of Operations — it lives
    # in the Cash Flow Statement. The tabular agent will miss, and the fallback
    # loop should automatically reroute to the Chroma text agent.
    run_query(
        "What was Apple's capital expenditure in 2025?",
        df, vector_store, embeddings, text_anchor, data_anchor,
    )

    # ── Interactive CLI ───────────────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("  INTERACTIVE MODE  |  type 'exit' to quit")
    print("#" * 70 + "\n")

    while True:
        try:
            user_input = input("  Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[CLI] Session terminated.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("[CLI] Goodbye.")
            break

        run_query(user_input, df, vector_store, embeddings, text_anchor, data_anchor)

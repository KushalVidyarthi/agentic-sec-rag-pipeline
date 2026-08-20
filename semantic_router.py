"""
semantic_router.py
------------------
Phase 3 -- Semantic Router: Intent-Based Query Dispatcher

Mathematically routes incoming user queries to one of two execution paths
by measuring cosine similarity between the query embedding and two
pre-defined "intent anchor" vectors:

  TEXT_INTENT  --> Chroma RAG retrieval (unstructured narrative text)
  DATA_INTENT  --> Pandas/SQL Execution Agent (structured financial tables)

Why cosine similarity?
  Embeddings from all-MiniLM-L6-v2 are normalized unit vectors (L2 norm = 1).
  For unit vectors, cosine similarity simplifies to the dot product:

      cos(θ) = (A · B) / (|A| * |B|)  →  A · B   (when |A| = |B| = 1)

  This gives a scalar in [-1, 1] where 1 = identical direction in embedding
  space. The intent anchor whose vector is most aligned with the query vector
  (highest dot product) represents the closest semantic intent.
"""

import warnings
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress the Chroma deprecation warning — langchain-chroma pulls in
# langchain-core>=1.5 which conflicts with langchain-community<1.0 pins.
# The community import remains stable for this pipeline version.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR: str = "./chroma_db"
CHROMA_COLLECTION: str = "sec_10k_text"

# Intent anchors — natural language descriptions of each routing target.
# These are embedded once at startup and reused for every query comparison.
# Anchor wording is deliberately broad to maximise coverage of synonymous
# phrasings a user might employ for each intent type.
TEXT_INTENT: str = (
    "Questions about company risks, future outlook, leadership, or textual summaries."
)
DATA_INTENT: str = (
    "Questions about revenue, balance sheets, calculations, or comparing financial numbers."
)

# Number of Chroma chunks to surface when TEXT_INTENT wins.
TOP_K_RESULTS: int = 2


# ─────────────────────────────────────────────
# INITIALISATION  (runs once at import time)
# ─────────────────────────────────────────────

print("[INIT] Loading embedding model...")
_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    # normalize_embeddings=True guarantees |v| = 1 for every vector produced,
    # which is the prerequisite for the dot-product cosine shortcut.
    encode_kwargs={"normalize_embeddings": True},
)

print("[INIT] Loading Chroma vector store from disk...")
_vector_store = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    collection_name=CHROMA_COLLECTION,
    embedding_function=_embeddings,
)

# Embed both intent anchors once.  Shape: (embedding_dim,) = (384,)
# np.array() wraps the list returned by embed_query into a numpy vector
# so we can use np.dot() directly.
print("[INIT] Embedding intent anchors...")
_text_anchor: np.ndarray = np.array(_embeddings.embed_query(TEXT_INTENT))
_data_anchor: np.ndarray = np.array(_embeddings.embed_query(DATA_INTENT))

print("[INIT] Router ready.\n")
print(f"  TEXT anchor : \"{TEXT_INTENT}\"")
print(f"  DATA anchor : \"{DATA_INTENT}\"")
print()


# ─────────────────────────────────────────────
# COSINE SIMILARITY HELPER
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D numpy vectors.

    Formula (general case):
        cos(θ) = dot(A, B) / (||A|| * ||B||)

    Because normalize_embeddings=True ensures ||A|| = ||B|| = 1, the
    denominator is always 1 and the expression collapses to np.dot(A, B).
    The general formula is retained here for correctness in case the
    embedding model or encode kwargs are changed in future phases.

    Returns
    -------
    float
        Scalar in [-1.0, 1.0]. Higher = more semantically similar.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        # Guard against zero-vector edge case (empty or whitespace-only query).
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ─────────────────────────────────────────────
# ROUTING FUNCTION
# ─────────────────────────────────────────────

def route_and_execute(user_query: str) -> None:
    """
    Embed *user_query*, compute similarity to both intent anchors, and
    dispatch to the appropriate execution path.

    Execution paths
    ---------------
    TEXT_INTENT wins  --> Chroma similarity_search, print top-K chunks.
    DATA_INTENT wins  --> Print analytical agent trigger message.

    Parameters
    ----------
    user_query : str
        The raw natural-language question from the user.
    """
    print("=" * 70)
    print(f"QUERY : {user_query}")
    print("=" * 70)

    # ── Step 1: Embed the query ───────────────────────────────────────────────
    # embed_query() returns a plain Python list; wrap in np.array for math ops.
    query_vec: np.ndarray = np.array(_embeddings.embed_query(user_query))

    # ── Step 2: Cosine similarity against both anchors ────────────────────────
    sim_text: float = cosine_similarity(query_vec, _text_anchor)
    sim_data: float = cosine_similarity(query_vec, _data_anchor)

    # Observability — print raw scores so engineers can tune anchor wording
    # or add new intent categories in future pipeline versions.
    print(f"[ROUTER] Similarity scores:")
    print(f"  TEXT_INTENT  : {sim_text:.6f}")
    print(f"  DATA_INTENT  : {sim_data:.6f}")

    # ── Step 3: Winner-takes-all dispatch ─────────────────────────────────────
    # The winning intent is simply the anchor with the higher cosine similarity.
    # The margin (delta) is printed for observability — a very small delta
    # (< 0.02) may indicate the query is ambiguous and warrants a hybrid path
    # in a future routing upgrade.
    delta = abs(sim_text - sim_data)
    winning_intent = "TEXT" if sim_text >= sim_data else "DATA"
    print(f"  Winner       : {winning_intent}_INTENT  (margin: {delta:.6f})")
    print()

    if sim_text >= sim_data:
        # ── TEXT path: semantic RAG retrieval ────────────────────────────────
        print(f"[ROUTER] Narrative intent detected. Querying Chroma (top-{TOP_K_RESULTS})...\n")
        results = _vector_store.similarity_search(user_query, k=TOP_K_RESULTS)

        for i, doc in enumerate(results, 1):
            # Truncate long chunks for readable terminal output while
            # preserving enough context to verify relevance.
            preview = doc.page_content.replace("\n", " ").strip()
            print(f"  [Chunk {i}]")
            print(f"  {preview[:400]}{'...' if len(preview) > 400 else ''}")
            print()

    else:
        # ── DATA path: analytical agent handoff ──────────────────────────────
        print("[ROUTER LOGIC] Analytical intent detected. "
              "Triggering Pandas/SQL Execution Agent...")
        print()


# ─────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1 — should resolve to TEXT_INTENT (risk / narrative language)
    route_and_execute("What are the major supply chain risks facing the company?")

    # Test 2 — should resolve to DATA_INTENT (quantitative / comparative language)
    route_and_execute("Did the gross margin for the services sector increase in 2025?")

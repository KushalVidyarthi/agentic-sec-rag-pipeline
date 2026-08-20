"""
build_vector_store.py
---------------------
Phase 2 -- Vector Engine: SEC 10-K Ingestion Pipeline

Pipeline stages:
  1. Text Extraction  -- pdfplumber reads every page; image-only pages are skipped.
  2. Chunking         -- RecursiveCharacterTextSplitter preserves semantic context
                         across chunk boundaries with a 200-token overlap.
  3. Embedding        -- all-MiniLM-L6-v2 via HuggingFaceEmbeddings (local, no API key).
  4. Persistence      -- Chroma writes vectors to ./chroma_db for reuse across sessions.
"""

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────────
# CONFIGURABLE VARIABLES
# ─────────────────────────────────────────────

# Source document
PDF_PATH: str = "apple_10k.pdf"

# Chunking parameters
# chunk_size=1000  → each chunk holds ~1000 characters of financial text, large
#                    enough to contain a full line item with its context.
# chunk_overlap=200 → 200-char overlap prevents a sentence from being split
#                     across two chunks and losing its semantic anchor.
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Embedding model — runs entirely locally via sentence-transformers.
# all-MiniLM-L6-v2 is a strong balance between speed and retrieval quality
# for domain-specific financial text without requiring an OpenAI API key.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Chroma persistence — vectors survive process restarts; no re-embedding needed.
CHROMA_PERSIST_DIR: str = "./chroma_db"
CHROMA_COLLECTION: str = "sec_10k_text"


# ─────────────────────────────────────────────
# STAGE 1: TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Iterate through every page of *pdf_path* and concatenate all extractable
    text into a single document string.

    Pages that return None from extract_text() are silently skipped — this
    covers image-only pages (charts, signature pages, cover art) that carry
    no textual information useful for retrieval.

    Returns
    -------
    str
        The full concatenated text of the document, with page boundaries
        delimited by double newlines for downstream chunking.
    """
    pages_extracted = 0
    pages_skipped = 0
    full_text_parts: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            if not text or not text.strip():
                # Image-only page or empty page — no text layer to extract.
                pages_skipped += 1
                continue

            full_text_parts.append(text.strip())
            pages_extracted += 1

    print(f"  [+] Pages extracted: {pages_extracted} / {total_pages}  |  Skipped (image/empty): {pages_skipped}")
    return "\n\n".join(full_text_parts)


# ─────────────────────────────────────────────
# STAGE 2: CHUNKING
# ─────────────────────────────────────────────

def chunk_text(full_text: str) -> list[str]:
    """
    Split *full_text* into overlapping chunks using RecursiveCharacterTextSplitter.

    The splitter attempts to break at paragraph boundaries first (\n\n), then
    sentence boundaries (\n), then word boundaries (space), and finally character
    level — preserving the most semantically coherent splits possible.

    Returns
    -------
    list[str]
        List of text chunk strings ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separators tried in priority order:
        #   \n\n → paragraph break (best for financial statement sections)
        #   \n   → line break (line items, table rows)
        #   " "  → word boundary (last resort before hard character split)
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(full_text)
    print(f"  [+] Total chunks created: {len(chunks)}")
    print(f"  [+] Avg chunk length: {sum(len(c) for c in chunks) // len(chunks)} chars")
    return chunks


# ─────────────────────────────────────────────
# STAGE 3 + 4: EMBEDDING + VECTOR STORE
# ─────────────────────────────────────────────

def build_vector_store(chunks: list[str]) -> Chroma:
    """
    Generate embeddings for *chunks* using a local HuggingFace sentence
    transformer and persist them in a Chroma collection on disk.

    Why HuggingFaceEmbeddings over OpenAIEmbeddings?
      - Zero API cost, no rate limits, fully reproducible across environments.
      - all-MiniLM-L6-v2 produces 384-dimensional dense vectors — efficient
        for local retrieval at 10-K scale (<10k chunks).

    Why Chroma with persist_directory?
      - Vectors written to disk survive process restarts — the expensive
        embedding step runs once and subsequent RAG queries load instantly.

    Returns
    -------
    Chroma
        The populated, persisted vector store instance.
    """
    print("  [+] Loading embedding model (downloads on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # Run inference on CPU by default — compatible with any machine.
        # Switch to {"device": "cuda"} if a GPU is available.
        model_kwargs={"device": "cpu"},
        # normalize_embeddings=True ensures cosine similarity == dot product,
        # which is the most stable metric for retrieval across chunk lengths.
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"  [+] Embedding model loaded: {EMBEDDING_MODEL}")

    print(f"  [+] Generating embeddings for {len(chunks)} chunks and writing to Chroma...")
    print(f"      Persist directory : {CHROMA_PERSIST_DIR}")
    print(f"      Collection name   : {CHROMA_COLLECTION}")

    # Chroma.from_texts() encodes all chunks in a single batched pass and
    # writes the resulting vectors + metadata to the local persist_directory.
    # This is idempotent if the collection already exists — re-running will
    # append to the existing collection rather than overwriting it.
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
    )

    print(f"  [+] Vector store persisted at: {CHROMA_PERSIST_DIR}")
    return vector_store


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Stage 1: Text Extraction ──────────────────────────────────────────────
    print("\n[STAGE 1/3] Extracting text from PDF...")
    full_text = extract_text_from_pdf(PDF_PATH)
    print(f"  [+] Total characters extracted: {len(full_text):,}")

    if not full_text.strip():
        raise RuntimeError(
            f"No text could be extracted from '{PDF_PATH}'.\n"
            "The PDF may be a scanned image document requiring OCR pre-processing."
        )

    # ── Stage 2: Chunking ─────────────────────────────────────────────────────
    print("\n[STAGE 2/3] Chunking document...")
    chunks = chunk_text(full_text)

    # ── Stage 3 + 4: Embedding + Persistence ─────────────────────────────────
    print("\n[STAGE 3/3] Generating embeddings and building vector store...")
    vector_store = build_vector_store(chunks)

    # ── Smoke test: verify retrieval works before declaring success ────────────
    print("\n[SMOKE TEST] Running a test similarity search...")
    test_query = "What was Apple's total net sales in 2025?"
    results = vector_store.similarity_search(test_query, k=3)

    print(f"  Query  : \"{test_query}\"")
    print(f"  Top-{len(results)} results retrieved:")
    for i, doc in enumerate(results, 1):
        # Show first 200 chars of each retrieved chunk for quick sanity check.
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i}] {preview}...")

    print("\n[SUCCESS] Vector store built successfully!")
    print(f"          Collection '{CHROMA_COLLECTION}' is ready for RAG queries.")
    print(f"          Load it in Phase 3 with:")
    print(f"          Chroma(persist_directory='{CHROMA_PERSIST_DIR}', "
          f"collection_name='{CHROMA_COLLECTION}', embedding_function=embeddings)")

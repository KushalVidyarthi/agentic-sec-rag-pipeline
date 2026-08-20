"""
extract_financials.py
---------------------
Phase 1 — Ingestion Engine: SEC 10-K Financial Table Extractor

Extracts the "Consolidated Statements of Operations" (or any dense financial
table) from a specific page of a PDF using pdfplumber's bounding-box-aware
table parser, then cleans and structures the result into a Pandas DataFrame.

Why pdfplumber over PyPDF2 / PyMuPDF / text loaders?
  - pdfplumber reconstructs the visual grid using actual character coordinates
    and inferred cell boundaries, preserving multi-column financial layouts.
  - Text-based extractors flatten the grid, causing column bleeding and broken
    row associations — fatal for numerical accuracy in financial pipelines.
"""

import pdfplumber
import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURABLE VARIABLES
# ─────────────────────────────────────────────

# Path to the target SEC 10-K PDF file.
pdf_path: str = "apple_10k.pdf"

# The exact phrase to search for when locating the target financial statement.
# Case-insensitive match — works even if the PDF uses mixed capitalisation.
TARGET_PHRASE: str = "CONSOLIDATED STATEMENTS OF OPERATIONS"


# ─────────────────────────────────────────────
# EXTRACTION SETTINGS
# ─────────────────────────────────────────────

# pdfplumber table extraction strategy — PRIMARY (drawn-border tables).
# "lines" detects table borders from actual vector lines in the PDF.
# Best for most modern SEC filings that use explicit grid borders.
TABLE_SETTINGS_LINES = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    # Snap / join tolerance in points — merges nearby or broken line segments
    # caused by minor rendering artifacts in PDF vector graphics.
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
}

# pdfplumber table extraction strategy — FALLBACK (whitespace-aligned tables).
# "text" infers column boundaries from character x-coordinates and row
# boundaries from text-line y-coordinates. Required for Apple 10-Ks and any
# filing that uses tab-stops / spaces instead of drawn grid lines.
TABLE_SETTINGS_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    # Tolerate larger horizontal gaps between text clusters before splitting
    # into separate columns — prevents over-segmentation of dollar amounts.
    "intersection_x_tolerance": 15,
    "intersection_y_tolerance": 3,
}


# ─────────────────────────────────────────────
# PAGE DISCOVERY
# ─────────────────────────────────────────────

def find_target_page(pdf_path: str, target_phrase: str) -> int:
    """
    Scan the PDF page-by-page and return the 0-indexed page number of the
    first page whose extracted text contains *target_phrase*.

    Strategy
    --------
    Uses pdfplumber's page.extract_text() which concatenates all character
    objects in reading order — reliable for text-layer PDFs like SEC filings.
    The comparison is case-insensitive and strips surrounding whitespace to
    guard against minor formatting differences across filing years.

    The loop breaks immediately on the first match (early-exit) so we never
    scan pages beyond the target — important for large 200+ page 10-K filings.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    target_phrase : str
        The heading text to search for (e.g. "CONSOLIDATED STATEMENTS OF OPERATIONS").

    Returns
    -------
    int
        0-indexed page number of the first matching page.

    Raises
    ------
    ValueError
        If the phrase is not found in any page — surfaces a clear pipeline error
        rather than silently passing None into downstream extraction.
    """
    phrase = target_phrase.strip().upper()  # normalise once before the loop

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # extract_text() returns the full text of the page as a single string,
            # or None if the page has no text layer (e.g. a pure-image scan).
            text = page.extract_text()

            if not text or phrase not in text.upper():
                continue

            # We must distinguish three types of page matches:
            #   1. TOC/index page  — phrase + trailing page number on the same line
            #   2. Prose reference — phrase embedded mid-sentence in the MD&A body text
            #   3. Actual heading  — phrase appears alone on its own line (what we want)
            #
            # Strategy: split the page text into lines and check each line that
            # contains the phrase. If the phrase occupies >60% of that line's
            # content, it is a heading-level match, not a prose reference.
            # TOC lines are caught because they include additional text (page numbers).
            lines = text.splitlines()
            is_heading_page = False
            for line in lines:
                if phrase in line.upper():
                    line_stripped = line.strip()
                    if len(line_stripped) == 0:
                        continue
                    # Ratio of the phrase length to the total line length.
                    # A standalone heading will be close to 1.0; a mid-sentence
                    # reference will be much lower (the line contains many more words).
                    dominance = len(phrase) / len(line_stripped)
                    if dominance >= 0.60:
                        is_heading_page = True
                        break

            if not is_heading_page:
                print(f"[INFO] Page {page_num} has phrase in prose/TOC context — skipping.")
                continue

            print(f"[INFO] Target phrase found as primary heading on page {page_num} (0-indexed).")
            return page_num


    # If we exit the loop without returning, the phrase was never found.
    raise ValueError(
        f"Phrase '{target_phrase}' not found in any page of '{pdf_path}'.\n"
        "  Check that:\n"
        "    - The PDF has a searchable text layer (not a scanned image).\n"
        "    - The spelling and capitalisation of TARGET_PHRASE exactly match\n"
        "      the statement heading in the filing."
    )


# ─────────────────────────────────────────────
# HELPER: Clean a raw cell value
# ─────────────────────────────────────────────

def _clean_cell(value) -> str | None:
    """
    Normalize a single cell extracted by pdfplumber.

    pdfplumber returns None for empty cells and may embed literal newline
    characters in multi-line header cells. This function:
      1. Passes None through unchanged (handled by pandas dropna later).
      2. Converts non-string types to str for safety.
      3. Replaces embedded newlines with a single space and strips edge whitespace.
    """
    if value is None:
        return None
    text = str(value)
    # Replace one or more consecutive newline sequences with a single space,
    # then collapse any resulting double-spaces and strip outer whitespace.
    return " ".join(text.splitlines()).strip() or None


# ─────────────────────────────────────────────
# MAIN EXTRACTION PIPELINE
# ─────────────────────────────────────────────

def extract_financial_table(pdf_path: str, target_page: int) -> pd.DataFrame | None:
    """
    Open *pdf_path*, navigate to *target_page*, extract the largest financial
    table matrix, and return it as a cleaned Pandas DataFrame.

    Extraction attempts two strategies in order:
      1. TABLE_SETTINGS_LINES — uses drawn border lines (fast, precise).
      2. TABLE_SETTINGS_TEXT  — infers columns from character positions
         (fallback for whitespace-aligned tables like Apple's 10-K).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with column headers derived from the first matrix row.
    None
        If no table is detected using either strategy.
    """
    with pdfplumber.open(pdf_path) as pdf:
        # Guard: target_page must be within document bounds.
        total_pages = len(pdf.pages)
        if target_page >= total_pages:
            raise IndexError(
                f"target_page={target_page} is out of range. "
                f"Document has {total_pages} page(s) (0-indexed: 0–{total_pages - 1})."
            )

        page = pdf.pages[target_page]

        # ── Strategy 1: drawn border lines ────────────────────────────────────
        # extract_table() returns the largest grid on the page as
        # List[List[str | None]], or None if no grid is detected.
        raw_matrix = page.extract_table(TABLE_SETTINGS_LINES)

        # ── Strategy 2: text-position inference (auto-fallback) ───────────────
        # Many SEC filings (incl. Apple) align columns with whitespace rather
        # than drawn borders. If lines strategy returns nothing, retry with
        # the text-based strategy before surfacing a warning.
        if not raw_matrix:
            print("[INFO] No border-line table found — retrying with text-position strategy...")
            raw_matrix = page.extract_table(TABLE_SETTINGS_TEXT)

    # ── Edge-case: no table detected ─────────────────────────────────────────
    if not raw_matrix:
        print(
            f"[WARNING] No table detected on page {target_page} of '{pdf_path}'.\n"
            "  Possible causes:\n"
            "    • The financial statement spans across pages — adjust target_page.\n"
            "    • The table uses text spacing instead of drawn borders — try\n"
            "      changing vertical_strategy/horizontal_strategy to 'text'.\n"
            "    • The page contains only images (scanned PDF) — OCR pre-processing required."
        )
        return None

    # ── Split header row from data rows ──────────────────────────────────────
    # SEC 10-K tables always use the first row for column labels (year headers,
    # metric names, etc.). Remaining rows are data records.
    raw_headers: list = raw_matrix[0]
    raw_rows: list[list] = raw_matrix[1:]

    # ── Clean headers ─────────────────────────────────────────────────────────
    # Multi-line column headers (e.g., "Net\nSales") are collapsed into a single
    # readable string. Duplicate header names are disambiguated with a suffix to
    # prevent silent column collisions in downstream joins.
    cleaned_headers: list[str] = []
    seen_headers: dict[str, int] = {}

    for raw_col in raw_headers:
        col = _clean_cell(raw_col) or "Unnamed"
        if col in seen_headers:
            seen_headers[col] += 1
            col = f"{col}_{seen_headers[col]}"
        else:
            seen_headers[col] = 0
        cleaned_headers.append(col)

    # ── Clean data rows ───────────────────────────────────────────────────────
    # Apply _clean_cell() to every cell to normalize newlines and whitespace.
    cleaned_rows = [
        [_clean_cell(cell) for cell in row]
        for row in raw_rows
    ]

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame(cleaned_rows, columns=cleaned_headers)

    # ── Drop entirely empty rows ──────────────────────────────────────────────
    # Rows where every cell is None (pdfplumber emits these for ruled separator
    # lines and blank spacer rows between statement sections).
    df.dropna(how="all", inplace=True)

    # ── Drop entirely empty columns ───────────────────────────────────────────
    # Columns where every value is None arise from phantom boundary lines or
    # merged-cell artifacts in the PDF's vector grid.
    df.dropna(axis=1, how="all", inplace=True)

    # ── Reset index after row drops ───────────────────────────────────────────
    # Preserves a clean 0-based RangeIndex for downstream iloc/loc operations.
    df.reset_index(drop=True, inplace=True)

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # -- Step 1: Dynamically locate the target statement page ------------------
    # Scans the PDF text layer, skipping TOC/index entries, and returns the
    # 0-indexed page number of the actual financial statement heading.
    print(f"[INFO] Scanning '{pdf_path}' for '{TARGET_PHRASE}'...")
    target_page = find_target_page(pdf_path, TARGET_PHRASE)

    # -- Step 2: Extract and clean the table from the discovered page ----------
    print(f"[INFO] Extracting table from page {target_page}...")
    df = extract_financial_table(pdf_path, target_page)

    if df is not None:
        print(f"\n[SUCCESS] Extracted table -- shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"[INFO] Columns detected: {list(df.columns)}\n")
        print("-" * 80)
        print("First 15 rows (structural integrity check):")
        print("-" * 80)
        # Suppress truncation for wide financial tables -- all columns visible.
        with pd.option_context(
            "display.max_columns", None,
            "display.max_colwidth", 50,
            "display.width", 200,
        ):
            print(df.head(15))
        print("-" * 80)
    else:
        print(
            "[ERROR] Could not extract a table from the target page.\n"
            "  Try adjusting TARGET_PHRASE or inspect the page range manually."
        )


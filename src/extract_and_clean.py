"""Extract and clean Google Prompt Engineering Whitepaper PDF into structured JSONL, manifest, and extraction log."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

PDF_PATH = Path("data/raw/Prompt Engineering.pdf")
OUTPUT_PAGES_PATH = Path("data/processed/pages.jsonl")
OUTPUT_MANIFEST_PATH = Path("data/manifest.json")
OUTPUT_LOG_PATH = Path("results/extraction_log.json")

DOCUMENT_ID = "google-prompt-engineering-2025"
DOCUMENT_TITLE = "Prompt Engineering"
DOCUMENT_AUTHOR = "Lee Boonstra"
DOCUMENT_VERSION = "February 2025"
DOCUMENT_YEAR = 2025

# Top ~9% and bottom ~10% margin ratios cleanly exclude the repeated
# "Prompt Engineering" header (y ~ 7.2%-8.8%) and "February 2025 | <page>"
# footer (y ~ 90.5%-92.1%) without intersecting body text (y ~ 16.5%-88.8%).
TOP_MARGIN_RATIO = 0.09
BOTTOM_MARGIN_RATIO = 0.10

# Canonical top-level sections mapped from document chapter headings
SECTION_HEADINGS: list[tuple[str, str]] = [
    ("Introduction", "Introduction"),
    ("LLM output configuration", "LLM Output Configuration"),
    ("Prompting techniques", "Prompting Techniques"),
    ("Code prompting", "Code Prompting"),
    ("Best Practices", "Best Practices"),
    ("Summary", "Summary"),
]


def normalize_text(text: str) -> str:
    """Normalize Unicode to NFC and clean whitespace while preserving paragraph breaks."""
    text = unicodedata.normalize("NFC", text)
    # Remove backspace/control chars (e.g. \x08) except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize non-breaking spaces and tabs
    text = text.replace("\u00a0", " ").replace("\t", " ")
    # Clean each line's trailing/excessive internal spaces
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.splitlines()]
    # Collapse 3+ blank lines into a single blank line
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def classify_non_content_page(page_num: int, raw_text: str, cleaned_text: str) -> str | None:
    """Return an exclusion reason if the page is non-content, else None."""
    if not cleaned_text:
        return "Trang trống hoặc không trích xuất được văn bản"

    first_lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()][:5]
    joined_head = " ".join(first_lines).lower()

    if page_num == 1 and "author:" in joined_head:
        return "Trang bìa (Cover page)"
    if first_lines and first_lines[0].lower() == "acknowledgements":
        return "Trang tác giả và lời cảm ơn (Acknowledgements)"
    if "\x08" in raw_text or (
        len(first_lines) >= 2
        and all(ln.isdigit() for ln in first_lines[1::2][:2])
        and len(cleaned_text) < 900
    ):
        return "Trang mục lục (Table of Contents)"
    if first_lines and first_lines[0].lower() == "endnotes":
        return "Trang tài liệu tham khảo / ghi chú cuối (Endnotes)"

    return None


def build_section_start_pages(doc: pymupdf.Document) -> dict[int, str]:
    """Read PDF TOC and map starting physical page numbers to canonical section names."""
    heading_lookup = {k.lower(): v for k, v in SECTION_HEADINGS}
    starts: dict[int, str] = {}
    for _level, title, page_num in doc.get_toc():
        canonical = heading_lookup.get(title.strip().lower())
        if canonical and page_num not in starts:
            starts[page_num] = canonical
    return starts


def extract_and_clean(
    pdf_path: Path = PDF_PATH,
    pages_out: Path = OUTPUT_PAGES_PATH,
    manifest_out: Path = OUTPUT_MANIFEST_PATH,
    log_out: Path = OUTPUT_LOG_PATH,
) -> None:
    """Extract content pages from PDF, assign section metadata, and write outputs."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    total_physical_pages = len(doc)
    section_starts = build_section_start_pages(doc)

    pages_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.parent.mkdir(parents=True, exist_ok=True)

    current_section = "Introduction"
    included_records: list[dict] = []
    excluded_records: list[dict] = []
    section_stats: dict[str, dict[str, int]] = {}

    for page_idx in range(total_physical_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        rect = page.rect
        clip = pymupdf.Rect(
            rect.x0,
            rect.y0 + rect.height * TOP_MARGIN_RATIO,
            rect.x1,
            rect.y1 - rect.height * BOTTOM_MARGIN_RATIO,
        )

        raw_text = page.get_text("text", clip=clip)
        cleaned_text = normalize_text(raw_text)

        if page_num in section_starts:
            current_section = section_starts[page_num]

        reason = classify_non_content_page(page_num, raw_text, cleaned_text)
        if reason is not None:
            preview = cleaned_text.replace("\n", " ")[:120]
            excluded_records.append(
                {
                    "page": page_num,
                    "reason": reason,
                    "char_count": len(cleaned_text),
                    "preview": preview,
                }
            )
            continue

        # Fallback: also update section if canonical heading appears as a standalone line
        for line in cleaned_text.splitlines():
            stripped = line.strip().lower()
            for heading_raw, canonical in SECTION_HEADINGS:
                if stripped == heading_raw.lower():
                    current_section = canonical
                    break

        record = {
            "document_id": DOCUMENT_ID,
            "source": pdf_path.as_posix(),
            "page": page_num,
            "section": current_section,
            "year": DOCUMENT_YEAR,
            "char_count": len(cleaned_text),
            "text": cleaned_text,
        }
        included_records.append(record)

        stat = section_stats.setdefault(current_section, {"pages": 0, "char_count": 0})
        stat["pages"] += 1
        stat["char_count"] += len(cleaned_text)

    # Write data/processed/pages.jsonl
    with pages_out.open("w", encoding="utf-8") as f:
        for rec in included_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write data/manifest.json
    total_chars = sum(r["char_count"] for r in included_records)
    manifest = {
        "source_id": DOCUMENT_ID,
        "title": DOCUMENT_TITLE,
        "author": DOCUMENT_AUTHOR,
        "source_path_or_url": pdf_path.as_posix(),
        "license_or_permission": "Public Google Whitepaper (February 2025), used for educational/research lab",
        "version_or_snapshot": DOCUMENT_VERSION,
        "accessed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_physical_pages": total_physical_pages,
        "included_page_count": len(included_records),
        "excluded_page_count": len(excluded_records),
        "included_pages": [r["page"] for r in included_records],
        "excluded_pages": [r["page"] for r in excluded_records],
        "page_ids": [f"{DOCUMENT_ID}:p{r['page']:03d}" for r in included_records],
        "total_characters": total_chars,
        "counting_method": (
            f"Physical PDF has {total_physical_pages} pages. Excluded non-content pages "
            "(cover, acknowledgements/authors, table of contents, endnotes). "
            f"Clipped top {int(TOP_MARGIN_RATIO * 100)}% and bottom {int(BOTTOM_MARGIN_RATIO * 100)}% "
            "margins via PyMuPDF to remove repeated header ('Prompt Engineering') and footer ('February 2025 | <page>')."
        ),
        "sections": section_stats,
        "pages": [
            {
                "page_id": f"{DOCUMENT_ID}:p{r['page']:03d}",
                "page": r["page"],
                "section": r["section"],
                "char_count": r["char_count"],
            }
            for r in included_records
        ],
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write results/extraction_log.json
    extraction_log = {
        "source_path": pdf_path.as_posix(),
        "total_physical_pages": total_physical_pages,
        "extracted_pages_count": len(included_records),
        "excluded_pages_count": len(excluded_records),
        "margin_clip_top_ratio": TOP_MARGIN_RATIO,
        "margin_clip_bottom_ratio": BOTTOM_MARGIN_RATIO,
        "excluded_pages": excluded_records,
        "section_summary": section_stats,
        "errors": [],
    }
    log_out.write_text(json.dumps(extraction_log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Extracted {len(included_records)}/{total_physical_pages} pages ({total_chars} chars).")
    print(f"Excluded {len(excluded_records)} pages: {[r['page'] for r in excluded_records]}")
    print(f"Sections: {section_stats}")


if __name__ == "__main__":
    extract_and_clean()

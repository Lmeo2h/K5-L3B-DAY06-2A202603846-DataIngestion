"""Independent verification script for benchmark/questions.json against data/processed/pages.jsonl."""

from __future__ import annotations

import json
import re
from pathlib import Path

PAGES_PATH = Path("data/processed/pages.jsonl")
QUESTIONS_PATH = Path("benchmark/questions.json")

REQUIRED_KEYS = {
    "id",
    "split",
    "question_type",
    "question",
    "expected_answer",
    "gold_source_id",
    "gold_page",
    "evidence_quote",
    "filter",
}
VALID_SPLITS = {"dev", "test"}
VALID_TYPES = {"truc_tiep", "dien_dat_lai", "dieu_kien_ngoai_le"}


def normalize_ws(text: str) -> str:
    """Collapse internal whitespace/newlines so line-wrapped PDF text matches cleanly."""
    return re.sub(r"\s+", " ", text).strip()


def verify() -> int:
    pages_by_key: dict[tuple[str, int], dict] = {}
    valid_sections: set[str] = set()

    with PAGES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pages_by_key[(rec["document_id"], rec["page"])] = rec
            if "section" in rec:
                valid_sections.add(rec["section"])

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []

    if len(questions) < 20:
        errors.append(f"Expected >= 20 questions, found {len(questions)}")

    split_counts = {"dev": 0, "test": 0}
    filter_counts = {"dev": 0, "test": 0}
    type_counts: dict[str, int] = {}
    seen_ids: set[str] = set()

    for idx, q in enumerate(questions, start=1):
        qid = q.get("id", f"index_{idx}")
        missing = REQUIRED_KEYS - set(q.keys())
        extra = set(q.keys()) - REQUIRED_KEYS
        if missing or extra:
            errors.append(f"[{qid}] Schema mismatch: missing={missing}, extra={extra}")
            continue

        if qid in seen_ids:
            errors.append(f"[{qid}] Duplicate question id")
        seen_ids.add(qid)

        split = q["split"]
        if split not in VALID_SPLITS:
            errors.append(f"[{qid}] Invalid split: {split}")
        else:
            split_counts[split] += 1

        qtype = q["question_type"]
        if qtype not in VALID_TYPES:
            errors.append(f"[{qid}] Invalid question_type: {qtype}")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

        gold_pages = q["gold_page"]
        if not isinstance(gold_pages, list) or not gold_pages or not all(isinstance(p, int) for p in gold_pages):
            errors.append(f"[{qid}] gold_page must be a non-empty list of ints: {gold_pages}")
            continue

        # Check no leakage of gold_source_id or gold_page in question or filter
        q_lower = q["question"].lower()
        if q["gold_source_id"].lower() in q_lower:
            errors.append(f"[{qid}] Leakage: gold_source_id appears in question")
        for pno in gold_pages:
            if re.search(rf"\b(?:page|trang)\s*{pno}\b", q_lower):
                errors.append(f"[{qid}] Leakage: gold_page {pno} appears in question")

        flt = q["filter"]
        if flt is not None:
            if not isinstance(flt, dict) or not flt:
                errors.append(f"[{qid}] filter must be null or non-empty dict")
            else:
                filter_counts[split] += 1
                if "section" in flt and flt["section"] not in valid_sections:
                    errors.append(f"[{qid}] filter section '{flt['section']}' not in corpus sections {sorted(valid_sections)}")
                if q["gold_source_id"] in flt.values() or any(p in flt.values() for p in gold_pages):
                    errors.append(f"[{qid}] Leakage: gold_source_id or gold_page in filter")

        # Verify evidence_quote exists verbatim in at least one gold_page and matches filter
        quote_norm = normalize_ws(q["evidence_quote"])
        matched_any_page = False
        for pno in gold_pages:
            rec = pages_by_key.get((q["gold_source_id"], pno))
            if rec is None:
                errors.append(f"[{qid}] (document_id={q['gold_source_id']}, page={pno}) not found in {PAGES_PATH}")
                continue
            page_text_norm = normalize_ws(rec["text"])
            if quote_norm in page_text_norm:
                matched_any_page = True
            if flt is not None:
                for k, v in flt.items():
                    if rec.get(k) != v:
                        errors.append(f"[{qid}] Filter {k}={v!r} does not match page {pno} metadata {rec.get(k)!r}")

        if not matched_any_page:
            errors.append(f"[{qid}] evidence_quote NOT found in gold_page {gold_pages}")

    if split_counts["dev"] != 15 or split_counts["test"] != 5:
        errors.append(f"Expected 15 dev / 5 test, got {split_counts}")
    if filter_counts["dev"] + filter_counts["test"] < 4 or filter_counts["dev"] < 3 or filter_counts["test"] < 1:
        errors.append(f"Filter distribution insufficient (need >=3 dev, >=1 test, >=4 total): {filter_counts}")

    print(f"Total questions : {len(questions)}")
    print(f"Splits          : {split_counts}")
    print(f"Question types  : {type_counts}")
    print(f"Filter counts   : {filter_counts}")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for e in errors:
            print(" -", e)
        return 1

    print("\nALL CHECKS PASSED (100% evidence & metadata verified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify())

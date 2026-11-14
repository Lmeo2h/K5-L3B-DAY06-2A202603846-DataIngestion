"""Ingest processed pages into ChromaDB using real BAAI/bge-m3 embeddings across 3 strategies."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

PAGES_PATH = Path("data/processed/pages.jsonl")
CHROMA_DIR = Path("data/chroma_db")
STATS_PATH = Path("results/ingestion_stats.json")
EXPERIMENT_LOG_PATH = Path("artifacts/experiment_log.csv")

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_PROVIDER = "HuggingFace / sentence-transformers (local)"
EMBEDDING_DIM = 1024


def load_pages(path: Path = PAGES_PATH) -> list[dict[str, Any]]:
    """Load cleaned pages from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Processed pages file not found: {path}")
    pages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    if not pages:
        raise ValueError(f"No pages found in {path}")
    return pages


def make_chunk_id(document_id: str, strategy: str, page: int, chunk_idx: int, text: str) -> str:
    """Create a deterministic chunk ID for idempotent upserts."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document_id}_{strategy}_p{page:03d}_c{chunk_idx:04d}_{digest}"


def chunk_fixed_size(
    pages: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 50,
    strategy: str = "A",
) -> list[dict[str, Any]]:
    """Strategy A: fixed-size character chunks with overlap."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []

    for rec in pages:
        text = rec["text"]
        if not text:
            continue
        start = 0
        idx = 0
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    {
                        "id": make_chunk_id(rec["document_id"], strategy, rec["page"], idx, piece),
                        "text": piece,
                        "metadata": {
                            "document_id": rec["document_id"],
                            "source": rec["source"],
                            "page": int(rec["page"]),
                            "section": rec["section"],
                            "year": int(rec["year"]),
                            "strategy": strategy,
                            "chunk_index": idx,
                        },
                    }
                )
                idx += 1
            if start + chunk_size >= len(text):
                break
            start += step
    return chunks


def _split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph along sentence boundaries (or hard cap)."""
    if len(para) <= max_chars:
        return [para]
    sentences = re.split(r"(?<=[.!?])\s+", para)
    parts: list[str] = []
    buf = ""
    for sent in sentences:
        if not sent:
            continue
        if len(sent) > max_chars:
            if buf:
                parts.append(buf.strip())
                buf = ""
            for i in range(0, len(sent), max_chars):
                parts.append(sent[i : i + max_chars].strip())
            continue
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                parts.append(buf.strip())
            buf = sent
    if buf:
        parts.append(buf.strip())
    return [p for p in parts if p]


def chunk_structure_aware(
    pages: list[dict[str, Any]],
    max_chars: int = 800,
    strategy: str = "B",
) -> list[dict[str, Any]]:
    """Strategy B/C: structure-aware chunking by headings/paragraphs up to max_chars."""
    chunks: list[dict[str, Any]] = []

    for rec in pages:
        raw_blocks = [b.strip() for b in re.split(r"\n\s*\n|\n", rec["text"]) if b.strip()]
        units: list[str] = []
        for block in raw_blocks:
            units.extend(_split_long_paragraph(block, max_chars))

        grouped: list[str] = []
        current = ""
        for unit in units:
            # Start a new chunk before short title-like headings when current buffer has content
            is_heading = (
                len(unit) <= 60
                and not unit.endswith((".", ",", ";", ":"))
                and unit[:1].isupper()
                and len(unit.split()) <= 8
            )
            if current and is_heading and len(current) >= 150:
                grouped.append(current.strip())
                current = unit
                continue

            candidate = f"{current}\n{unit}".strip() if current else unit
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    grouped.append(current.strip())
                current = unit
        if current:
            grouped.append(current.strip())

        for idx, piece in enumerate(grouped):
            chunks.append(
                {
                    "id": make_chunk_id(rec["document_id"], strategy, rec["page"], idx, piece),
                    "text": piece,
                    "metadata": {
                        "document_id": rec["document_id"],
                        "source": rec["source"],
                        "page": int(rec["page"]),
                        "section": rec["section"],
                        "year": int(rec["year"]),
                        "strategy": strategy,
                        "chunk_index": idx,
                    },
                }
            )
    return chunks


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Encode texts into normalized dense vectors; raise on failure (no mock fallback)."""
    if not texts:
        raise ValueError("Cannot encode empty text list")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = embeddings.tolist()
    if len(vectors) != len(texts) or len(vectors[0]) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected embedding output shape: got {len(vectors)}x{len(vectors[0]) if vectors else 0}"
        )
    return vectors


def upsert_collection_idempotent(
    client: chromadb.PersistentClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    batch_size: int = 100,
) -> dict[str, Any]:
    """Upsert chunks deterministically and remove any stale IDs no longer in the corpus."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    new_ids = [c["id"] for c in chunks]
    new_id_set = set(new_ids)

    existing = collection.get(include=[])
    existing_ids = set(existing.get("ids", []))
    stale_ids = list(existing_ids - new_id_set)
    if stale_ids:
        collection.delete(ids=stale_ids)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=vectors[i : i + batch_size],
        )

    count_after_first = collection.count()

    # Re-ingest a verification batch (all chunks) to prove idempotency (no duplicates)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=vectors[i : i + batch_size],
        )
    count_after_reingest = collection.count()

    if count_after_first != len(chunks) or count_after_reingest != count_after_first:
        raise RuntimeError(
            f"Idempotency check failed for {collection_name}: "
            f"expected={len(chunks)}, first={count_after_first}, reingest={count_after_reingest}"
        )

    return {
        "collection_name": collection_name,
        "stale_deleted": len(stale_ids),
        "before_reingest_count": count_after_first,
        "after_reingest_count": count_after_reingest,
        "no_duplicates": count_after_first == count_after_reingest,
    }


def compute_chunk_stats(chunks: list[dict[str, Any]], elapsed_sec: float) -> dict[str, Any]:
    """Compute chunk count, length distribution, total chars, and estimated tokens."""
    lengths = [len(c["text"]) for c in chunks]
    total_chars = sum(lengths)
    return {
        "chunk_count": len(chunks),
        "char_length_min": min(lengths),
        "char_length_max": max(lengths),
        "char_length_avg": round(total_chars / len(chunks), 2),
        "total_chars": total_chars,
        "estimated_tokens": round(total_chars / 4),
        "ingest_time_seconds": round(elapsed_sec, 3),
        "estimated_cost_usd": 0.0,
    }


def update_experiment_log(stats: dict[str, Any]) -> None:
    """Update artifacts/experiment_log.csv with ingestion metadata for strategies A, B, C."""
    rows = [
        {
            "strategy": "A",
            "chunking": "fixed_size",
            "chunk_size": "500",
            "overlap": "50",
            "metadata": "document_id,source,page,section,year",
            "filter": "none",
            "vectorizer": f"{EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM}d, cosine)",
            "hypothesis": "Baseline fixed-size chunking; may split sentences/concepts across boundaries.",
            "dev_hit1": "",
            "dev_hit3": "",
            "test_hit1": "",
            "test_hit3": "",
            "notes": f"chunks={stats['A']['chunk_count']}, avg_len={stats['A']['char_length_avg']}, time={stats['A']['ingest_time_seconds']}s",
        },
        {
            "strategy": "B",
            "chunking": "structure_aware",
            "chunk_size": "max_800",
            "overlap": "0",
            "metadata": "document_id,source,page,section,year",
            "filter": "none",
            "vectorizer": f"{EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM}d, cosine)",
            "hypothesis": "Paragraph/heading-aware chunking improves coherence and evidence completeness over A.",
            "dev_hit1": "",
            "dev_hit3": "",
            "test_hit1": "",
            "test_hit3": "",
            "notes": f"chunks={stats['B']['chunk_count']}, avg_len={stats['B']['char_length_avg']}, time={stats['B']['ingest_time_seconds']}s",
        },
        {
            "strategy": "C",
            "chunking": "structure_aware",
            "chunk_size": "max_800",
            "overlap": "0",
            "metadata": "document_id,source,page,section,year",
            "filter": "section (when filter != null)",
            "vectorizer": f"{EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM}d, cosine)",
            "hypothesis": "Applying section metadata filter narrows search space and improves precision on filtered queries.",
            "dev_hit1": "",
            "dev_hit3": "",
            "test_hit1": "",
            "test_hit3": "",
            "notes": f"chunks={stats['C']['chunk_count']}, avg_len={stats['C']['char_length_avg']}, time={stats['C']['ingest_time_seconds']}s",
        },
    ]
    fieldnames = list(rows[0].keys())
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ingest() -> None:
    """Run full ingestion pipeline for Strategies A, B, and C."""
    pages = load_pages()
    print(f"Loaded {len(pages)} cleaned pages from {PAGES_PATH}.")

    print(f"Loading real embedding model: {EMBEDDING_MODEL_NAME} ...")
    model_t0 = time.perf_counter()
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model_load_sec = time.perf_counter() - model_t0
    print(f"Model loaded in {model_load_sec:.2f}s.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Strategy A: Fixed-size 500 chars, overlap 50
    t0 = time.perf_counter()
    chunks_a = chunk_fixed_size(pages, chunk_size=500, overlap=50, strategy="A")
    vectors_a = encode_texts(model, [c["text"] for c in chunks_a])
    idem_a = upsert_collection_idempotent(client, "strategy_a", chunks_a, vectors_a)
    elapsed_a = time.perf_counter() - t0
    stats_a = {**compute_chunk_stats(chunks_a, elapsed_a), "idempotency_check": idem_a}
    print(f"[Strategy A] {stats_a['chunk_count']} chunks, idempotent={idem_a['no_duplicates']}, time={stats_a['ingest_time_seconds']}s")

    # Strategy B: Structure-aware max 800 chars
    t0 = time.perf_counter()
    chunks_b = chunk_structure_aware(pages, max_chars=800, strategy="B")
    vectors_b = encode_texts(model, [c["text"] for c in chunks_b])
    idem_b = upsert_collection_idempotent(client, "strategy_b", chunks_b, vectors_b)
    elapsed_b = time.perf_counter() - t0
    stats_b = {**compute_chunk_stats(chunks_b, elapsed_b), "idempotency_check": idem_b}
    print(f"[Strategy B] {stats_b['chunk_count']} chunks, idempotent={idem_b['no_duplicates']}, time={stats_b['ingest_time_seconds']}s")

    # Strategy C: Same chunks as Strategy B, stored with metadata for filtered retrieval
    t0 = time.perf_counter()
    chunks_c = [
        {
            "id": make_chunk_id(c["metadata"]["document_id"], "C", c["metadata"]["page"], c["metadata"]["chunk_index"], c["text"]),
            "text": c["text"],
            "metadata": {**c["metadata"], "strategy": "C"},
        }
        for c in chunks_b
    ]
    idem_c = upsert_collection_idempotent(client, "strategy_c", chunks_c, vectors_b)
    elapsed_c = time.perf_counter() - t0
    stats_c = {**compute_chunk_stats(chunks_c, elapsed_c), "idempotency_check": idem_c}
    print(f"[Strategy C] {stats_c['chunk_count']} chunks (reused B chunks/vectors), idempotent={idem_c['no_duplicates']}, time={stats_c['ingest_time_seconds']}s")

    all_stats = {
        "embedding_model": {
            "provider": EMBEDDING_PROVIDER,
            "model_name": EMBEDDING_MODEL_NAME,
            "dimension": EMBEDDING_DIM,
            "normalize_embeddings": True,
            "model_load_seconds": round(model_load_sec, 3),
            "cost_usd": 0.0,
        },
        "vector_store": {
            "engine": "ChromaDB",
            "version": chromadb.__version__,
            "path": CHROMA_DIR.as_posix(),
            "metric": "cosine",
            "search_type": "HNSW approximate nearest neighbor (cosine)",
        },
        "strategies": {
            "A": stats_a,
            "B": stats_b,
            "C": stats_c,
        },
    }

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    update_experiment_log(all_stats["strategies"])
    print(f"Saved ingestion stats to {STATS_PATH} and updated {EXPERIMENT_LOG_PATH}.")


if __name__ == "__main__":
    run_ingest()

"""Evaluate Strategies A, B, C on dev and test splits and generate traces, metrics, and manual audit templates."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

QUESTIONS_PATH = Path("benchmark/questions.json")
PAGES_PATH = Path("data/processed/pages.jsonl")
CHROMA_DIR = Path("data/chroma_db")
TRACES_DIR = Path("results/traces")
METRICS_PATH = Path("results/metrics_summary.json")
AUDIT_DIR = Path("results/manual_audit")
EXPERIMENT_LOG_PATH = Path("artifacts/experiment_log.csv")
INGESTION_STATS_PATH = Path("results/ingestion_stats.json")

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

STRATEGIES = {
    "A": {"collection": "strategy_a", "use_filter": False},
    "B": {"collection": "strategy_b", "use_filter": False},
    "C": {"collection": "strategy_c", "use_filter": True},
}


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_chunk_hit(chunk_text: str, metadata: dict[str, Any], question: dict[str, Any]) -> bool:
    """Return True if chunk matches gold_source_id, gold_page, and contains evidence_quote."""
    if metadata.get("document_id") != question["gold_source_id"]:
        return False
    if metadata.get("page") not in question["gold_page"]:
        return False
    return normalize_ws(question["evidence_quote"]) in normalize_ws(chunk_text)


def is_filter_compliant(metadatas: list[dict[str, Any]], flt: dict[str, Any] | None) -> bool | None:
    """Check whether all top-k chunks satisfy the metadata filter."""
    if not flt:
        return None
    if not metadatas:
        return False
    for meta in metadatas:
        for k, v in flt.items():
            if meta.get(k) != v:
                return False
    return True


def load_embedding_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


def compute_split_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    hit1 = sum(1 for r in records if r["hit_at_1"])
    hit3 = sum(1 for r in records if r["hit_at_3"])
    filtered = [r for r in records if r["filter_applied"] is not None]
    if filtered:
        compliant = sum(1 for r in filtered if r["filter_compliant"] is True)
        flt_metric = {
            "compliant": compliant,
            "total": len(filtered),
            "ratio_str": f"{compliant}/{len(filtered)}",
            "percent": round(100.0 * compliant / len(filtered), 2),
        }
    else:
        flt_metric = {
            "compliant": 0,
            "total": 0,
            "ratio_str": "N/A",
            "percent": None,
        }
    return {
        "n_questions": total,
        "hit_at_1": {
            "hits": hit1,
            "total": total,
            "ratio_str": f"{hit1}/{total}",
            "percent": round(100.0 * hit1 / total, 2) if total else 0.0,
        },
        "evidence_hit_at_3": {
            "hits": hit3,
            "total": total,
            "ratio_str": f"{hit3}/{total}",
            "percent": round(100.0 * hit3 / total, 2) if total else 0.0,
        },
        "filter_compliance": flt_metric,
    }


def generate_coherence_audit(client: chromadb.PersistentClient) -> Path:
    """Create manual coherence audit template for the first 10 sorted page IDs."""
    pages = [json.loads(line) for line in PAGES_PATH.open("r", encoding="utf-8") if line.strip()]
    pages.sort(key=lambda r: r["page"])
    first_10 = pages[:10]

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIT_DIR / "coherence_audit.csv"
    fieldnames = [
        "page_id",
        "page",
        "section",
        "strategy",
        "chunk_id",
        "chunk_length",
        "chunk_text",
        "coherence_score_0_to_2",
        "reason",
    ]
    rows: list[dict[str, Any]] = []

    collections = {s: client.get_collection(cfg["collection"]) for s, cfg in STRATEGIES.items()}

    for p in first_10:
        page_num = int(p["page"])
        page_id = f"{p['document_id']}:p{page_num:03d}"
        for strat in ["A", "B", "C"]:
            res = collections[strat].get(
                where={"$and": [{"page": page_num}, {"chunk_index": 0}]},
                include=["documents", "metadatas"],
            )
            if res["ids"]:
                cid = res["ids"][0]
                text = res["documents"][0]
            else:
                cid = ""
                text = ""
            rows.append(
                {
                    "page_id": page_id,
                    "page": page_num,
                    "section": p["section"],
                    "strategy": strat,
                    "chunk_id": cid,
                    "chunk_length": len(text),
                    "chunk_text": text,
                    "coherence_score_0_to_2": "",
                    "reason": "",
                }
            )

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def generate_grounding_and_metadata_audit(traces: dict[str, list[dict[str, Any]]]) -> Path:
    """Create manual grounding & metadata audit template for the 5 test questions."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIT_DIR / "grounding_and_metadata_audit.csv"
    fieldnames = [
        "strategy",
        "question_id",
        "question_type",
        "question",
        "expected_answer",
        "gold_page",
        "evidence_quote",
        "filter_applied",
        "rank",
        "chunk_id",
        "chunk_page",
        "chunk_section",
        "chunk_source",
        "cosine_score",
        "auto_hit",
        "chunk_text",
        "metadata_complete_and_correct_0_or_1",
        "grounding_score_0_to_2",
        "reason",
    ]
    rows: list[dict[str, Any]] = []
    for strat in ["A", "B", "C"]:
        test_items = [t for t in traces[strat] if t["split"] == "test"]
        for item in test_items:
            for chunk in item["top_3"]:
                meta = chunk["metadata"]
                rows.append(
                    {
                        "strategy": strat,
                        "question_id": item["id"],
                        "question_type": item["question_type"],
                        "question": item["question"],
                        "expected_answer": item["expected_answer"],
                        "gold_page": str(item["gold_page"]),
                        "evidence_quote": item["evidence_quote"],
                        "filter_applied": json.dumps(item["filter_applied"], ensure_ascii=False)
                        if item["filter_applied"]
                        else "none",
                        "rank": chunk["rank"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_page": meta.get("page"),
                        "chunk_section": meta.get("section"),
                        "chunk_source": meta.get("source"),
                        "cosine_score": chunk["cosine_score"],
                        "auto_hit": chunk["is_hit"],
                        "chunk_text": chunk["text"],
                        "metadata_complete_and_correct_0_or_1": "",
                        "grounding_score_0_to_2": "",
                        "reason": "",
                    }
                )

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def update_experiment_log_metrics(summary: dict[str, Any]) -> None:
    """Fill dev/test Hit@1 and Hit@3 columns in artifacts/experiment_log.csv."""
    if not EXPERIMENT_LOG_PATH.exists():
        return
    with EXPERIMENT_LOG_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    for row in rows:
        strat = row["strategy"]
        if strat in summary["strategies"]:
            s_metrics = summary["strategies"][strat]
            row["dev_hit1"] = s_metrics["dev"]["hit_at_1"]["ratio_str"]
            row["dev_hit3"] = s_metrics["dev"]["evidence_hit_at_3"]["ratio_str"]
            row["test_hit1"] = s_metrics["test"]["hit_at_1"]["ratio_str"]
            row["test_hit3"] = s_metrics["test"]["evidence_hit_at_3"]["ratio_str"]

    fieldnames = list(rows[0].keys())
    with EXPERIMENT_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}.")

    model = load_embedding_model()
    query_texts = [q["question"] for q in questions]
    query_vectors = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    all_traces: dict[str, list[dict[str, Any]]] = {}
    metrics_summary: dict[str, Any] = {"strategies": {}}

    for strat, cfg in STRATEGIES.items():
        collection = client.get_collection(cfg["collection"])
        strat_traces: list[dict[str, Any]] = []

        for q, qvec in zip(questions, query_vectors):
            flt = q["filter"] if (cfg["use_filter"] and q["filter"] is not None) else None
            res = collection.query(
                query_embeddings=[qvec],
                n_results=3,
                where=flt,
                include=["documents", "metadatas", "distances"],
            )

            ids = res["ids"][0]
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]

            top_3: list[dict[str, Any]] = []
            for rank, (cid, text, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
                hit = is_chunk_hit(text, meta, q)
                top_3.append(
                    {
                        "rank": rank,
                        "chunk_id": cid,
                        "cosine_score": round(1.0 - float(dist), 6),
                        "distance": round(float(dist), 6),
                        "metadata": meta,
                        "text": text,
                        "is_hit": hit,
                    }
                )

            hit1 = top_3[0]["is_hit"] if top_3 else False
            hit3 = any(c["is_hit"] for c in top_3)
            flt_comp = is_filter_compliant(metas, flt)

            strat_traces.append(
                {
                    "id": q["id"],
                    "split": q["split"],
                    "question_type": q["question_type"],
                    "question": q["question"],
                    "expected_answer": q["expected_answer"],
                    "filter_applied": flt,
                    "gold_source_id": q["gold_source_id"],
                    "gold_page": q["gold_page"],
                    "evidence_quote": q["evidence_quote"],
                    "hit_at_1": hit1,
                    "hit_at_3": hit3,
                    "filter_compliant": flt_comp,
                    "top_3": top_3,
                }
            )

        trace_path = TRACES_DIR / f"trace_{strat}.json"
        trace_path.write_text(json.dumps(strat_traces, ensure_ascii=False, indent=2), encoding="utf-8")
        all_traces[strat] = strat_traces

        dev_recs = [r for r in strat_traces if r["split"] == "dev"]
        test_recs = [r for r in strat_traces if r["split"] == "test"]
        metrics_summary["strategies"][strat] = {
            "dev": compute_split_metrics(dev_recs),
            "test": compute_split_metrics(test_recs),
            "overall": compute_split_metrics(strat_traces),
        }

        d = metrics_summary["strategies"][strat]["dev"]
        t = metrics_summary["strategies"][strat]["test"]
        print(
            f"[Strategy {strat}] "
            f"DEV Hit@1={d['hit_at_1']['ratio_str']} ({d['hit_at_1']['percent']}%), "
            f"Hit@3={d['evidence_hit_at_3']['ratio_str']} ({d['evidence_hit_at_3']['percent']}%), "
            f"Filter={d['filter_compliance']['ratio_str']} | "
            f"TEST Hit@1={t['hit_at_1']['ratio_str']} ({t['hit_at_1']['percent']}%), "
            f"Hit@3={t['evidence_hit_at_3']['ratio_str']} ({t['evidence_hit_at_3']['percent']}%), "
            f"Filter={t['filter_compliance']['ratio_str']}"
        )

    METRICS_PATH.write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    coherence_path = generate_coherence_audit(client)
    grounding_path = generate_grounding_and_metadata_audit(all_traces)
    update_experiment_log_metrics(metrics_summary)

    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved manual audit templates to {coherence_path} and {grounding_path}")


if __name__ == "__main__":
    run_evaluation()

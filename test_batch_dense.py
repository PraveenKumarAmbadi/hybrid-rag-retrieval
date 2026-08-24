#!/usr/bin/env python3
"""Batch correctness test for dense retrieval.

Runs dense search across all 300 test queries and compares against qrels.
Reports Recall@K, queries with hits, and average best rank.
"""

import logging
from pathlib import Path

from configs.loader import load_config
from indexing.dense_encoder import DenseEncoder
from indexing.faiss_indexer import FaissIndex
from ingestion.loaders import load_corpus, load_qrels, load_queries
from retrieval.dense import search_dense

logger = logging.getLogger(__name__)


def run_batch_test(top_k: int = 100) -> dict:
    """Run dense retrieval across all test queries and report metrics.

    Args:
        top_k: Number of results to retrieve per query.

    Returns:
        Stats dict with batch test metrics.
    """
    config = load_config()
    dense_cfg = config["dense"]

    # Load data
    corpus = load_corpus(Path(config["data"]["corpus_path"]))
    queries = load_queries(Path(config["data"]["queries_path"]))
    qrels = load_qrels(Path(config["data"]["qrels_path"]))

    # Load FAISS index
    logger.info("Loading FAISS index ...")
    faiss_index = FaissIndex.load(
        dense_cfg["index_path"],
        dense_cfg["ids_path"],
    )

    # Initialize encoder once
    encoder = DenseEncoder()

    # Test only on queries that have qrels
    test_query_ids = set(qrels.keys())
    logger.info(
        "Running dense retrieval on %d test queries (top_k=%d) ...",
        len(test_query_ids),
        top_k,
    )

    queries_with_hits = 0
    total_relevant = 0
    total_found = 0
    best_ranks: list[int] = []

    for qid in test_query_ids:
        query = queries[qid]
        relevant_docs = set(qrels[qid].keys())

        results = search_dense(
            query_text=query.text,
            faiss_index=faiss_index,
            corpus=corpus,
            encoder=encoder,
            top_k=top_k,
        )

        found_doc_ids = {r.doc_id for r in results}
        hits = relevant_docs & found_doc_ids

        if hits:
            queries_with_hits += 1

            # Best rank: lowest position (1-indexed) of any relevant doc
            rank_map = {r.doc_id: i + 1 for i, r in enumerate(results)}
            best_rank = min(rank_map[did] for did in hits)
            best_ranks.append(best_rank)

        total_relevant += len(relevant_docs)
        total_found += len(hits)

    recall = total_found / total_relevant if total_relevant > 0 else 0.0
    avg_best_rank = sum(best_ranks) / len(best_ranks) if best_ranks else float("inf")

    stats = {
        "total_test_queries": len(test_query_ids),
        "queries_with_hits": queries_with_hits,
        "queries_with_zero_hits": len(test_query_ids) - queries_with_hits,
        "hit_rate": round(queries_with_hits / len(test_query_ids), 4),
        "total_relevant_docs": total_relevant,
        "total_found_docs": total_found,
        "recall_at_k": round(recall, 4),
        "average_best_rank": round(avg_best_rank, 2) if best_ranks else None,
        "top_k": top_k,
    }

    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = run_batch_test(top_k=100)

    print("\n" + "=" * 50)
    print("DENSE BATCH TEST RESULTS")
    print("=" * 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()

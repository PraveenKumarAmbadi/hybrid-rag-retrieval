#!/usr/bin/env python3
"""Batch correctness test for hybrid retrieval (BM25 + SPLADE + dense + RRF).

Runs hybrid search across all 300 test queries and compares against qrels.
Also runs each individual method for head-to-head comparison.
"""

import logging
from pathlib import Path

from configs.loader import load_config
from indexing.dense_encoder import DenseEncoder
from indexing.es_client import get_es_client
from indexing.faiss_indexer import FaissIndex
from ingestion.loaders import load_corpus, load_qrels, load_queries
from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from retrieval.fusion import fuse_rrf
from retrieval.splade import search_splade

logger = logging.getLogger(__name__)


def _compute_metrics(results, relevant_docs):
    """Compute hit count and best rank for a single query."""
    found_doc_ids = {r.doc_id for r in results}
    hits = relevant_docs & found_doc_ids

    best_rank = None
    if hits:
        rank_map = {r.doc_id: i + 1 for i, r in enumerate(results)}
        best_rank = min(rank_map[did] for did in hits)

    return len(hits), best_rank


def run_batch_test(top_k: int = 100) -> dict:
    """Run hybrid retrieval across all test queries and report metrics.

    Also runs each individual method for comparison.

    Args:
        top_k: Number of results to retrieve per query.

    Returns:
        Stats dict with batch test metrics for hybrid and individual methods.
    """
    config = load_config()
    data_cfg = config["data"]
    dense_cfg = config["dense"]
    fusion_cfg = config["fusion"]

    # Load data
    corpus = load_corpus(Path(data_cfg["corpus_path"]))
    queries = load_queries(Path(data_cfg["queries_path"]))
    qrels = load_qrels(Path(data_cfg["qrels_path"]))

    # Load infrastructure once
    logger.info("Loading FAISS index ...")
    faiss_index = FaissIndex.load(
        dense_cfg["index_path"],
        dense_cfg["ids_path"],
    )

    logger.info("Loading dense encoder ...")
    dense_encoder = DenseEncoder()

    logger.info("Connecting to Elasticsearch ...")
    es_client = get_es_client()

    test_query_ids = sorted(qrels.keys())
    logger.info(
        "Running hybrid + individual retrieval on %d test queries (top_k=%d) ...",
        len(test_query_ids),
        top_k,
    )

    # Accumulators for each method
    methods = {
        "bm25": {"hits": 0, "found": 0, "best_ranks": []},
        "splade": {"hits": 0, "found": 0, "best_ranks": []},
        "dense": {"hits": 0, "found": 0, "best_ranks": []},
        "hybrid": {"hits": 0, "found": 0, "best_ranks": []},
    }

    total_relevant = 0

    for qid in test_query_ids:
        query = queries[qid]
        relevant_docs = set(qrels[qid].keys())
        total_relevant += len(relevant_docs)

        # BM25
        bm25_results = search_bm25(
            client=es_client,
            index_name="scifact_bm25",
            query_text=query.text,
            top_k=top_k,
            fields=["full_text"],
        )
        n_found, best_rank = _compute_metrics(bm25_results, relevant_docs)
        methods["bm25"]["found"] += n_found
        if best_rank:
            methods["bm25"]["hits"] += 1
            methods["bm25"]["best_ranks"].append(best_rank)

        # SPLADE
        splade_results = search_splade(
            query_text=query.text,
            index_name="scifact_splade",
            top_k=top_k,
        )
        n_found, best_rank = _compute_metrics(splade_results, relevant_docs)
        methods["splade"]["found"] += n_found
        if best_rank:
            methods["splade"]["hits"] += 1
            methods["splade"]["best_ranks"].append(best_rank)

        # Dense
        dense_results = search_dense(
            query_text=query.text,
            faiss_index=faiss_index,
            corpus=corpus,
            encoder=dense_encoder,
            top_k=top_k,
        )
        n_found, best_rank = _compute_metrics(dense_results, relevant_docs)
        methods["dense"]["found"] += n_found
        if best_rank:
            methods["dense"]["hits"] += 1
            methods["dense"]["best_ranks"].append(best_rank)

        # Hybrid (RRF fusion of the three lists above)
        fused_results = fuse_rrf(
            ranked_lists=[bm25_results, splade_results, dense_results],
            k=fusion_cfg["rrf_k"],
        )[:top_k]
        n_found, best_rank = _compute_metrics(fused_results, relevant_docs)
        methods["hybrid"]["found"] += n_found
        if best_rank:
            methods["hybrid"]["hits"] += 1
            methods["hybrid"]["best_ranks"].append(best_rank)

    # Build stats dict
    stats = {
        "total_test_queries": len(test_query_ids),
        "total_relevant_docs": total_relevant,
        "top_k": top_k,
    }

    for method_name, data in methods.items():
        hit_rate = data["hits"] / len(test_query_ids) if test_query_ids else 0.0
        recall = data["found"] / total_relevant if total_relevant > 0 else 0.0
        avg_best_rank = (
            sum(data["best_ranks"]) / len(data["best_ranks"])
            if data["best_ranks"]
            else None
        )

        stats[method_name] = {
            "queries_with_hits": data["hits"],
            "queries_with_zero_hits": len(test_query_ids) - data["hits"],
            "hit_rate": round(hit_rate, 4),
            "total_found_docs": data["found"],
            "recall_at_k": round(recall, 4),
            "average_best_rank": round(avg_best_rank, 2) if avg_best_rank else None,
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

    print("\n" + "=" * 60)
    print("HYBRID BATCH TEST RESULTS")
    print("=" * 60)
    print(f"  Total test queries: {stats['total_test_queries']}")
    print(f"  Total relevant docs: {stats['total_relevant_docs']}")
    print(f"  top_k: {stats['top_k']}")
    print("-" * 60)

    for method in ["bm25", "splade", "dense", "hybrid"]:
        m = stats[method]
        print(f"\n  {method.upper()}:")
        print(f"    Queries with >=1 hit: {m['queries_with_hits']}/{stats['total_test_queries']} ({m['hit_rate']*100:.1f}%)")
        print(f"    Recall@{stats['top_k']}: {m['total_found_docs']}/{stats['total_relevant_docs']} ({m['recall_at_k']*100:.1f}%)")
        print(f"    Avg best rank: {m['average_best_rank']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

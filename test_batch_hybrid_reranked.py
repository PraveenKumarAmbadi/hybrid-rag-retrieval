#!/usr/bin/env python3
"""Batch test: Hybrid retrieval vs Hybrid + Cross-Encoder reranking.

Runs hybrid search across all 300 test queries, then reranks the top-100
results with a cross-encoder.  Compares metrics head-to-head.
"""

import logging
from pathlib import Path

from configs.loader import load_config
from indexing.dense_encoder import DenseEncoder
from indexing.es_client import get_es_client
from indexing.faiss_indexer import FaissIndex
from ingestion.loaders import load_corpus, load_qrels, load_queries
from reranking.cross_encoder import CrossEncoderReranker
from retrieval.fusion import fuse_rrf
from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from retrieval.splade import search_splade
from retrieval.types import FusedResult, RerankedResult

logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging: your code talks, libraries stay quiet."""
    # 1. Root handler — controls the output format
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 2. Quiet down chatty third-party loggers
    noisy_loggers = [
        "elastic_transport",   # ES HTTP request logs
        "elasticsearch",       # Older ES client logs
        "urllib3",             # HTTP connection pool logs
        "requests",            # If you use requests elsewhere
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

def _compute_metrics(results, relevant_docs, top_k: int = 100):
    """Compute hit count and best rank for a single query."""
    found_doc_ids = {r.doc_id for r in results[:top_k]}
    hits = relevant_docs & found_doc_ids

    best_rank = None
    if hits:
        rank_map = {r.doc_id: i + 1 for i, r in enumerate(results)}
        best_rank = min(rank_map[did] for did in hits)

    return len(hits), best_rank


def run_batch_test(top_k: int = 100) -> dict:
    """Run hybrid + reranked hybrid across all test queries."""
    config = load_config()
    data_cfg = config["data"]
    dense_cfg = config["dense"]
    fusion_cfg = config["fusion"]
    reranker_cfg = config["reranker"]

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

    logger.info("Loading cross-encoder reranker ...")
    reranker = CrossEncoderReranker(
    model_name=reranker_cfg["model_name"],
    max_seq_length=reranker_cfg["max_seq_length"],
    device=reranker_cfg.get("device"),
    )

    test_query_ids = sorted(qrels.keys())
    logger.info(
        "Running hybrid vs hybrid+rerank on %d queries (top_k=%d) ...",
        len(test_query_ids),
        top_k,
    )

    methods = {
        "hybrid": {"hits": 0, "found": 0, "best_ranks": []},
        "hybrid_reranked": {"hits": 0, "found": 0, "best_ranks": []},
    }

    total_relevant = 0

    for qid in test_query_ids:
        query = queries[qid]
        relevant_docs = set(qrels[qid].keys())
        total_relevant += len(relevant_docs)

        # --- Stage 1: Hybrid retrieval (same as Phase 5) ---
        bm25_results = search_bm25(
            client=es_client,
            index_name="scifact_bm25",
            query_text=query.text,
            top_k=100,
            fields=["full_text"],
        )

        splade_results = search_splade(
            query_text=query.text,
            index_name="scifact_splade",
            top_k=100,
        )

        dense_results = search_dense(
            query_text=query.text,
            faiss_index=faiss_index,
            corpus=corpus,
            encoder=dense_encoder,
            top_k=100,
        )

        fused_results = fuse_rrf(
            ranked_lists=[bm25_results, splade_results, dense_results],
            k=fusion_cfg["rrf_k"],
        )[:top_k]

        # --- Stage 2: Cross-encoder reranking ---
        reranked_results = reranker.rerank(
            query_text=query.text,
            results=fused_results,
            batch_size=reranker_cfg["batch_size"],
            top_k=top_k,
        )

        # Metrics for hybrid (before rerank)
        n_found, best_rank = _compute_metrics(fused_results, relevant_docs, top_k)
        methods["hybrid"]["found"] += n_found
        if best_rank:
            methods["hybrid"]["hits"] += 1
            methods["hybrid"]["best_ranks"].append(best_rank)

        # Metrics for hybrid + rerank
        n_found_r, best_rank_r = _compute_metrics(reranked_results, relevant_docs, top_k)
        methods["hybrid_reranked"]["found"] += n_found_r
        if best_rank_r:
            methods["hybrid_reranked"]["hits"] += 1
            methods["hybrid_reranked"]["best_ranks"].append(best_rank_r)

    # Build stats
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
    print("HYBRID vs HYBRID + RERANK BATCH TEST RESULTS")
    print("=" * 60)
    print(f"  Total test queries: {stats['total_test_queries']}")
    print(f"  Total relevant docs: {stats['total_relevant_docs']}")
    print(f"  top_k: {stats['top_k']}")
    print("-" * 60)

    for method in ["hybrid", "hybrid_reranked"]:
        m = stats[method]
        print(f"\n  {method.upper().replace('_', ' + ')}:")
        print(f"    Queries with >=1 hit: {m['queries_with_hits']}/{stats['total_test_queries']} ({m['hit_rate']*100:.1f}%)")
        print(f"    Recall@{stats['top_k']}: {m['total_found_docs']}/{stats['total_relevant_docs']} ({m['recall_at_k']*100:.1f}%)")
        print(f"    Avg best rank: {m['average_best_rank']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    setup_logging(logging.INFO)
    main()
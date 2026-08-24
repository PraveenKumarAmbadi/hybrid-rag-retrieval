#!/usr/bin/env python3
"""Unified evaluation benchmark: all retrieval methods, all standard IR metrics.
Runs BM25, SPLADE, Dense, Hybrid (RRF), and Hybrid+BGE across all 300
SciFact test queries.  Computes Recall@K, Precision@K, nDCG@K, MRR@K,
and MAP@K for K in {1, 5, 10, 20, 50, 100}.
Usage::
python evaluate_all.py
Output: a formatted comparison table printed to stdout.
"""
from pathlib import Path
from typing import Dict, List, Tuple
from configs.loader import load_config
from evaluation.runner import Evaluator
from indexing.dense_encoder import DenseEncoder
from indexing.es_client import get_es_client
from indexing.faiss_indexer import FaissIndex
from indexing.splade_encoder import SpladeEncoder
from ingestion.loaders import load_corpus, load_qrels, load_queries
from reranking.cross_encoder import CrossEncoderReranker
from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from retrieval.fusion import fuse_rrf
from retrieval.splade import search_splade
from setup_logging import setup_logging

logger = setup_logging()


def _extract_doc_ids(results) -> List[str]:
    """Pull doc_ids out of any result type (SearchResult, FusedResult, RerankedResult)."""
    return [r.doc_id for r in results]


def run_all_methods(top_k: int = 100) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, int]]]:
    """Run every retrieval method on every test query.
    
    Returns::
        (runs, qrels) where:
        runs = {
            "bm25": {query_id: [doc_id, ...], ...},
            "splade": {...},
            "dense": {...},
            "hybrid": {...},
            "hybrid_bge": {...},
        }
        qrels = {query_id: {doc_id: relevance_score, ...}, ...}
    """
    config = load_config()
    data_cfg = config["data"]
    bm25_cfg = config["bm25"]
    splade_cfg = config["splade"]
    dense_cfg = config["dense"]
    fusion_cfg = config["fusion"]
    reranker_cfg = config["reranker"]

    # Load data
    logger.info("Loading corpus, queries, qrels ...")
    corpus = load_corpus(Path(data_cfg["corpus_path"]))
    queries = load_queries(Path(data_cfg["queries_path"]))
    qrels = load_qrels(Path(data_cfg["qrels_path"]))

    # Load infrastructure once
    logger.info("Loading FAISS index ...")
    faiss_index = FaissIndex.load(dense_cfg["index_path"], dense_cfg["ids_path"])
    logger.info("Loading dense encoder ...")
    dense_encoder = DenseEncoder()
    logger.info("Loading SPLADE encoder ...")
    splade_encoder = SpladeEncoder()
    logger.info("Connecting to Elasticsearch ...")
    es_client = get_es_client()
    logger.info("Loading BGE reranker ...")
    reranker = CrossEncoderReranker(
        model_name=reranker_cfg["model_name"],
        max_seq_length=reranker_cfg["max_seq_length"],
    )

    test_query_ids = sorted(qrels.keys())
    logger.info("Evaluating %d queries across 5 methods (top_k=%d) ...", len(test_query_ids), top_k)

    # Storage for raw results per method per query
    runs: Dict[str, Dict[str, List[str]]] = {
        "bm25": {},
        "splade": {},
        "dense": {},
        "hybrid": {},
        "hybrid_bge": {},
    }

    for idx, qid in enumerate(test_query_ids, start=1):
        if idx % 50 == 0:
            logger.info("Progress: %d/%d queries", idx, len(test_query_ids))
        
        query = queries[qid]

        # --- BM25 ---
        bm25_results = search_bm25(
            client=es_client,
            index_name=bm25_cfg["index_name"],
            query_text=query.text,
            top_k=top_k,
            fields=["full_text"],
        )
        runs["bm25"][qid] = _extract_doc_ids(bm25_results)

        # --- SPLADE ---
        splade_results = search_splade(
            query_text=query.text,
            index_name=splade_cfg["index_name"],
            top_k=top_k,
        )
        runs["splade"][qid] = _extract_doc_ids(splade_results)

        # --- Dense ---a
        dense_results = search_dense(
            query_text=query.text,
            faiss_index=faiss_index,
            corpus=corpus,
            encoder=dense_encoder,
            top_k=top_k,
        )
        runs["dense"][qid] = _extract_doc_ids(dense_results)

        # --- Hybrid (RRF) ---
        fused_results = fuse_rrf(
            ranked_lists=[bm25_results, splade_results, dense_results],
            k=fusion_cfg["rrf_k"],
        )[:top_k]
        runs["hybrid"][qid] = _extract_doc_ids(fused_results)

        # --- Hybrid + BGE Rerank ---
        reranked_results = reranker.rerank(
            query_text=query.text,
            results=fused_results,
            batch_size=reranker_cfg["batch_size"],
            top_k=top_k,
        )
        runs["hybrid_bge"][qid] = _extract_doc_ids(reranked_results)

    return runs, qrels


def _print_table(report: Dict[str, Dict[str, Dict[int, float]]], k_values: List[int]) -> None:
    """Pretty-print the evaluation report."""
    metrics = ["recall", "precision", "ndcg", "mrr", "map"]
    methods = ["bm25", "splade", "dense", "hybrid", "hybrid_bge"]
    method_labels = {
        "bm25": "BM25",
        "splade": "SPLADE",
        "dense": "Dense",
        "hybrid": "Hybrid (RRF)",
        "hybrid_bge": "Hybrid + BGE",
    }

    for metric in metrics:
        print(f"\n{'=' * 70}")
        print(f"  {metric.upper()}@K")
        print(f"{'=' * 70}")

        # Header: 18 chars for Method, 9 chars per K column
        header = f"{'Method':<18}"
        for k in k_values:
            header += f"{'@K=' + str(k):<9}"
        print(f"  {header}")
        
        # Divider matching the exact width
        print(f"  {'-' * (18 + 9 * len(k_values))}")

        # Rows: 18 chars for Method, 9 chars per K column (right-aligned)
        for method in methods:
            row = f"{method_labels[method]:<18}"
            for k in k_values:
                val = report[method][metric].get(k, 0.0)
                row += f"{val:>9.4f}"
            print(f"  {row}")


def main() -> None:
    logger = setup_logging()

    k_values = [1, 5, 10, 20, 50, 100]
    runs, qrels = run_all_methods(top_k=100)

    # Evaluate each method
    reports: Dict[str, Dict[str, Dict[int, float]]] = {}
    for method_name, method_runs in runs.items():
        evaluator = Evaluator(qrels)
        for qid, ranked_doc_ids in method_runs.items():
            evaluator.add_run(qid, ranked_doc_ids)
        reports[method_name] = evaluator.aggregate(k_values=k_values)

    # Print
    print("\n" + "=" * 70)
    print("  UNIFIED EVALUATION BENCHMARK — SciFact Test Split (300 queries)")
    print("=" * 70)
    _print_table(reports, k_values)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
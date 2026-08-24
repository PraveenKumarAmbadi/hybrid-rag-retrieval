#!/usr/bin/env python3
"""Batch correctness test for SPLADE retrieval against SciFact qrels.

Runs SPLADE search across the test queries and reports:
- Queries with >=1 relevant doc found
- Overall Recall@K
- Average best rank of found relevant docs
"""

import logging
from pathlib import Path

from configs.loader import load_config
from ingestion.loaders import load_queries, load_qrels
from retrieval.splade import search_splade

logger = logging.getLogger(__name__)


def run_batch_test(top_k: int = 100) -> None:
    """Run SPLADE batch retrieval test against test qrels."""
    config = load_config()

    queries = load_queries(Path(config["data"]["queries_path"]))
    qrels = load_qrels(Path(config["data"]["qrels_path"]))

    # Only test queries that have ground truth in qrels
    test_query_ids = set(qrels.keys())
    logger.info("Testing %d queries with ground truth ...", len(test_query_ids))

    queries_with_hits = 0
    total_relevant = 0
    total_found = 0
    rank_sum = 0
    rank_count = 0

    for qid in test_query_ids:
        query = queries.get(qid)
        if not query:
            continue

        relevant_docs = set(qrels[qid].keys())
        total_relevant += len(relevant_docs)

        results = search_splade(query.text, top_k=top_k)

        found_ids = {r.doc_id for r in results}
        found_relevant = relevant_docs & found_ids

        if found_relevant:
            queries_with_hits += 1

        total_found += len(found_relevant)

        # Best rank among found relevant docs
        for rank, r in enumerate(results, start=1):
            if r.doc_id in relevant_docs:
                rank_sum += rank
                rank_count += 1
                break  # Only count best rank per query

    num_tested = len(test_query_ids)

    print("\n" + "=" * 55)
    print("SPLADE BATCH TEST RESULTS")
    print(f"  top_k={top_k}, queries tested={num_tested}")
    print("=" * 55)
    print(f"  Queries with >=1 hit: {queries_with_hits}/{num_tested} "
          f"({queries_with_hits / num_tested * 100:.1f}%)")
    print(f"  Overall Recall@{top_k}: {total_found}/{total_relevant} "
          f"({total_found / total_relevant * 100:.1f}%)")
    if rank_count > 0:
        print(f"  Average best rank: {rank_sum / rank_count:.1f}")
    print("=" * 55)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_batch_test()

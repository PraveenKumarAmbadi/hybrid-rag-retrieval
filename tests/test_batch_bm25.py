#!/usr/bin/env python3
"""Batch correctness test: BM25 performance across multiple queries."""

from pathlib import Path

from configs.loader import load_config
from indexing.es_client import get_es_client
from ingestion.loaders import load_queries, load_qrels
from retrieval.bm25 import search_bm25


def main(fields:list[str]=['full_text'], top_k: int = 100) -> None:
    config = load_config()
    index_name = config["elasticsearch"]["index_name"]

    queries = load_queries(Path(config["data"]["queries_path"]))
    qrels = load_qrels(Path(config["data"]["qrels_path"]))

    # Test first N queries with qrels. 50 is a quick sample; set to None for all.
    max_queries = 300
    qids_to_test = list(qrels.keys())[:max_queries]

    client = get_es_client()

    total_queries = 0
    queries_with_hits = 0
    total_relevant = 0
    total_found = 0
    best_ranks: list[int] = []

    for qid in qids_to_test:
        if qid not in queries:
            continue

        query = queries[qid]
        relevant_docs = qrels[qid]
        num_relevant = len(relevant_docs)

        results = search_bm25(
            client,
            index_name,
            query.text,
            top_k=top_k,
            fields=fields,
        )

        rank_lookup = {r.doc_id: rank for rank, r in enumerate(results, start=1)}

        found = 0
        best_rank: int | None = None

        for doc_id in relevant_docs:
            if doc_id in rank_lookup:
                found += 1
                rank = rank_lookup[doc_id]
                if best_rank is None or rank < best_rank:
                    best_rank = rank

        total_queries += 1
        total_relevant += num_relevant
        total_found += found

        if found > 0:
            queries_with_hits += 1
            best_ranks.append(best_rank)

    print("=" * 50)
    print("BATCH BM25 CORRECTNESS TEST")
    print("=" * 50)
    print(f"Fields searched:          {fields}")    
    print(f"Queries tested:           {total_queries}")
    print(f"Queries with ≥1 hit:      {queries_with_hits} ({queries_with_hits / total_queries * 100:.1f}%)")
    print(f"Total relevant docs:      {total_relevant}")
    print(f"Total found in top-100:   {total_found}")
    print(f"Overall Recall@100:       {total_found / total_relevant * 100:.1f}%")
    if best_ranks:
        print(f"Average best rank:        {sum(best_ranks) / len(best_ranks):.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main(fields=['full_text'], top_k=100)  # Change to ["title", "text"] to search multiple fields  
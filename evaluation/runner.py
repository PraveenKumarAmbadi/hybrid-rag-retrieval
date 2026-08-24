"""Evaluation runner — orchestrates metric computation across all queries.

Usage::
    evaluator = Evaluator(qrels)
    for query_id, ranked_doc_ids in results:
        evaluator.add_run(query_id, ranked_doc_ids)
    report = evaluator.aggregate(k_values=[1, 5, 10, 20, 50, 100])
"""

import logging
from typing import Dict, List, Set

from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    reciprocal_rank,
    average_precision_at_k,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Accumulates ranked results per query and computes aggregate metrics.

    The evaluator is stateful: you add one query's ranked results at a time,
    then call ``aggregate()`` to get mean metrics across all queries.
    """

    def __init__(self, qrels: Dict[str, Dict[str, int]]) -> None:
        """Initialise with ground-truth relevance judgments.

        Args:
            qrels: ``{query_id: {doc_id: relevance_score}}``.
                For SciFact, relevance scores are binary (0 or 1).
        """
        self.qrels = qrels
        self._runs: Dict[str, List[str]] = {}

    def add_run(self, query_id: str, ranked_doc_ids: List[str]) -> None:
        """Record the ranked document IDs for a single query.

        Args:
            query_id: The query identifier (must exist in qrels).
            ranked_doc_ids: List of document IDs in ranking order
                (best first).  May contain duplicates — they are ignored
                after the first occurrence.
        """
        if query_id not in self.qrels:
            logger.warning("Query %s not found in qrels — skipping.", query_id)
            return

        # Deduplicate while preserving order (first occurrence wins)
        seen: Set[str] = set()
        deduped: List[str] = []
        for did in ranked_doc_ids:
            if did not in seen:
                seen.add(did)
                deduped.append(did)

        self._runs[query_id] = deduped

    def aggregate(
        self,
        k_values: List[int] = None,
    ) -> Dict[str, Dict[int, float]]:
        """Compute mean metrics across all recorded queries.

        Returns a nested dict::

            {
                "recall":     {1: 0.45, 5: 0.72, ...},
                "precision":  {1: 0.45, 5: 0.18, ...},
                "ndcg":       {1: 0.45, 5: 0.62, ...},
                "mrr":        {1: 0.45, 5: 0.58, ...},   # MRR@K
                "map":        {1: 0.45, 5: 0.55, ...},   # MAP@K
            }

        MRR@K uses the first relevant doc within top-K; if none, RR=0.
        MAP@K is average precision computed only within top-K.
        """
        if k_values is None:
            k_values = [1, 5, 10, 20, 50, 100]

        # Collect per-query scores for each metric at each K
        scores: Dict[str, Dict[int, List[float]]] = {
            "recall": {k: [] for k in k_values},
            "precision": {k: [] for k in k_values},
            "ndcg": {k: [] for k in k_values},
            "mrr": {k: [] for k in k_values},
            "map": {k: [] for k in k_values},
        }

        for qid, ranked_doc_ids in self._runs.items():
            relevant_docs = {did for did, rel in self.qrels[qid].items() if rel > 0}

            for k in k_values:
                scores["recall"][k].append(recall_at_k(ranked_doc_ids, relevant_docs, k))
                scores["precision"][k].append(precision_at_k(ranked_doc_ids, relevant_docs, k))
                scores["ndcg"][k].append(ndcg_at_k(ranked_doc_ids, relevant_docs, k))
                scores["mrr"][k].append(reciprocal_rank(ranked_doc_ids[:k], relevant_docs))
                scores["map"][k].append(average_precision_at_k(ranked_doc_ids, relevant_docs, k))

        # Compute means
        report: Dict[str, Dict[int, float]] = {}
        for metric_name, k_scores in scores.items():
            report[metric_name] = {
                k: round(sum(vals) / len(vals), 4) if vals else 0.0
                for k, vals in k_scores.items()
            }

        return report


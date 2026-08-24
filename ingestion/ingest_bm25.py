#!/usr/bin/env python3
"""BM25-specific corpus ingestion pipeline.

This script ingests raw SciFact documents into an Elasticsearch BM25 index.
It assumes the index already exists — index creation is a separate admin step.

Orchestrates: load → bulk index → refresh → verify.
"""

import logging
from pathlib import Path

from configs.loader import load_config
from indexing.es_client import get_es_client
from indexing.indexer import bulk_index_documents
from ingestion.loaders import load_corpus


logger = logging.getLogger(__name__)


def ingest_corpus(
    corpus_path: str | None = None,
    index_name: str | None = None,
    batch_size: int = 500,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> dict:
    """Ingest a corpus into an Elasticsearch BM25 index.

    Args:
        corpus_path: Path to corpus JSONL. Defaults to config['data']['corpus_path'].
        index_name: Target ES index. Defaults to config['elasticsearch']['index_name'].
        batch_size: Documents per bulk request chunk.
        max_retries: Max retry attempts on transient failures.
        backoff_seconds: Base delay for exponential backoff.

    Returns:
        Stats dict with ingestion metrics.

    Raises:
        RuntimeError: if the target index does not exist.
    """
    config = load_config()

    if corpus_path is None:
        corpus_path = config["data"]["corpus_path"]
    if index_name is None:
        index_name = config["elasticsearch"]["index_name"]

    corpus_path = Path(corpus_path)

    # Phase 1: Load
    logger.info("Loading corpus from %s ...", corpus_path)
    documents = load_corpus(corpus_path)
    logger.info("Loaded %d documents.", len(documents))

    # Phase 2: Connect and verify index exists
    client = get_es_client()
    if not client.indices.exists(index=index_name):
        raise RuntimeError(
            f"Index '{index_name}' does not exist. "
            f"Create it first with: python create_bm25_index.py"
        )
    logger.info("Index '%s' exists. Ready to ingest.", index_name)

    # Phase 3: Bulk index (delegates to indexer.py)
    stats = bulk_index_documents(
        client,
        index_name,
        documents,
        batch_size=batch_size,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )

    # Phase 4: Refresh and verify count
    client.indices.refresh(index=index_name)

    actual_count = client.count(index=index_name)["count"]
    expected_count = len(documents)

    if actual_count != expected_count:
        logger.warning(
            "Count mismatch: expected %d, found %d.",
            expected_count,
            actual_count,
        )
    else:
        logger.info(
            "Count verified: %d documents in index '%s'.",
            actual_count,
            index_name,
        )

    stats["index_count"] = actual_count
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = ingest_corpus()

    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    for key, value in stats.items():p
    
    print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()
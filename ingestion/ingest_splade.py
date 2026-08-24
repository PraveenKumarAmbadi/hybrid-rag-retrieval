#!/usr/bin/env python3
"""SPLADE-specific corpus ingestion pipeline.

Loads the SciFact corpus, encodes each document's full_text into a SPLADE
sparse vector, and bulk-indexes into Elasticsearch.

This is NOT a generic ingestion script — BM25 and dense retrieval use
different pipelines.
"""

import logging
import time
from pathlib import Path

from elasticsearch import ConnectionError, ConnectionTimeout, TransportError
from elasticsearch.helpers import bulk

from configs.loader import load_config
from indexing.es_client import get_es_client
from indexing.splade_encoder import SpladeEncoder
from ingestion.loaders import Document, load_corpus

logger = logging.getLogger(__name__)


def _splade_doc_to_action(
    index_name: str, doc: Document, sparse_vec: dict[int, float]
) -> dict:
    """Convert a Document + SPLADE vector into an ES bulk action dict.

    Elasticsearch sparse_vector fields expect string keys, so token IDs
    (integers) are converted to strings.
    """
    sparse_as_str_keys = {str(k): v for k, v in sparse_vec.items()}
    return {
        "_index": index_name,
        "_id": doc.doc_id,
        "_source": {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "text": doc.text,
            "sparse_vector": sparse_as_str_keys,
        },
    }


def _is_retryable(exc: Exception) -> bool:
    """Return True only for transient failures worth retrying.

    Non-retryable exceptions (BadRequestError, AuthenticationException,
    NotFoundError, etc.) propagate immediately — fail fast.
    """
    if isinstance(exc, (ConnectionError, ConnectionTimeout)):
        return True
    if isinstance(exc, TransportError):
        status = getattr(exc, "status_code", None) or getattr(
            exc, "meta", {}
        ).get("status", 0)
        return status in (429, 503, 504)
    return False


def ingest_splade_corpus(
    corpus_path: str | None = None,
    index_name: str | None = None,
    batch_size: int = 500,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> dict:
    """Ingest a corpus into an Elasticsearch SPLADE index.

    Args:
        corpus_path: Path to corpus JSONL. Defaults to config.
        index_name: Target ES index. Defaults to config.
        batch_size: Documents per bulk request chunk.
        max_retries: Max retry attempts on transient failures.
        backoff_seconds: Base delay for exponential backoff.

    Returns:
        Stats dict with ingestion metrics.
    """
    config = load_config()

    if corpus_path is None:
        corpus_path = config["data"]["corpus_path"]
    if index_name is None:
        index_name = config["splade"]["index_name"]

    corpus_path = Path(corpus_path)

    # Phase 1: Load
    logger.info("Loading corpus from %s ...", corpus_path)
    documents = load_corpus(corpus_path)
    logger.info("Loaded %d documents.", len(documents))

    # Phase 2: Connect & verify index exists
    client = get_es_client()
    logger.info("Connected to Elasticsearch.")

    if not client.indices.exists(index=index_name):
        raise RuntimeError(
            f"Index '{index_name}' does not exist. "
            f"Create it first with: python create_splade_index.py"
        )

    # Phase 3: Encode all documents with SPLADE
    logger.info("Encoding %d documents with SPLADE ...", len(documents))
    encoder = SpladeEncoder()

    doc_list = list(documents.values())
    texts = [doc.full_text for doc in doc_list]

    start_encode = time.time()
    sparse_vectors = encoder.encode_texts(texts)
    encode_time = time.time() - start_encode
    logger.info(
        "Encoding complete: %d vectors in %.1fs (%.1f docs/sec).",
        len(sparse_vectors),
        encode_time,
        len(sparse_vectors) / encode_time,
    )

    # Phase 4: Build bulk actions
    actions = [
        _splade_doc_to_action(index_name, doc, vec)
        for doc, vec in zip(doc_list, sparse_vectors)
    ]
    logger.info("Prepared %d bulk actions.", len(actions))

    # Phase 5: Bulk index with retry
    success_count = 0
    error_list = []
    retries = 0

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "Bulk indexing (attempt %d/%d) ...",
                attempt + 1,
                max_retries + 1,
            )
            success_count, error_list = bulk(
                client,
                actions,
                chunk_size=batch_size,
                raise_on_error=False,
            )
            logger.info(
                "Bulk complete: %d succeeded, %d failed.",
                success_count,
                len(error_list),
            )
            break

        except Exception as exc:
            if _is_retryable(exc) and attempt < max_retries:
                wait = backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Transient failure (%s: %s). Retrying in %.1fs ...",
                    type(exc).__name__,
                    exc,
                    wait,
                )
                time.sleep(wait)
                retries += 1
            else:
                if not _is_retryable(exc):
                    logger.error(
                        "Non-retryable failure (%s: %s). Aborting.",
                        type(exc).__name__,
                        exc,
                    )
                else:
                    logger.error(
                        "Bulk failed after %d retries: %s",
                        max_retries,
                        exc,
                    )
                raise

    # Log individual errors (first 5 only)
    if error_list:
        logger.warning("%d document(s) failed indexing:", len(error_list))
        for err in error_list[:5]:
            logger.warning("  %s", err)
        if len(error_list) > 5:
            logger.warning("  ... and %d more.", len(error_list) - 5)

    # Phase 6: Refresh and verify
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

    return {
        "documents_loaded": len(documents),
        "documents_encoded": len(sparse_vectors),
        "documents_indexed": success_count,
        "failed_documents": len(error_list),
        "retries": retries,
        "index_count": actual_count,
        "encode_time_seconds": round(encode_time, 2),
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = ingest_splade_corpus()

    print("\n" + "=" * 50)
    print("SPLADE INGESTION SUMMARY")
    print("=" * 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()
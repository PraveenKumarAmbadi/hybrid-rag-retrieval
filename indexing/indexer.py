"""Create Elasticsearch BM25 index and bulk-ingest documents."""

import logging
import time
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, ConnectionTimeout
from elasticsearch.helpers import bulk

from indexing.mapping import SCIFACT_BM25_MAPPING
from ingestion.loaders import Document


logger = logging.getLogger(__name__)


def create_index(
    client: Elasticsearch,
    index_name: str,
    recreate: bool = False,
) -> None:
    """Create the SciFact BM25 index with the explicit mapping.

    Args:
        client: An active Elasticsearch client.
        index_name: Name of the index to create.
        recreate: If True, delete and recreate the index if it already exists.
                  If False (default), skip creation if the index exists.
    """
    exists = client.indices.exists(index=index_name)

    if exists and not recreate:
        logger.info("Index '%s' already exists. Skipping creation.", index_name)
        return

    if exists and recreate:
        logger.warning("Index '%s' exists — deleting (recreate=True).", index_name)
        client.indices.delete(index=index_name)

    client.indices.create(index=index_name, mappings=SCIFACT_BM25_MAPPING)
    logger.info("Index '%s' created.", index_name)


def _doc_to_action(index_name: str, doc: Document) -> dict[str, Any]:
    """Convert a Document into an Elasticsearch bulk action dict.

    Note: full_text is NOT included in _source — Elasticsearch builds it
    via copy_to at index time from title + text, avoiding storage duplication.
    """
    return {
        "_index": index_name,
        "_id": doc.doc_id,
        "_source": {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "text": doc.text,
        },
    }


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception warrants a retry.

    Retryable: connection failures and transient HTTP errors from ES
    (429 Too Many Requests, 503 Service Unavailable, 504 Gateway Timeout).
    Non-retryable: BadRequestError (400), NotFoundError (404), auth failures, etc.
    """
    if isinstance(exc, (ConnectionError, ConnectionTimeout)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 503, 504):
        return True
    return False


def bulk_index_documents(
    client: Elasticsearch,
    index_name: str,
    documents: dict[str, Document],
    batch_size: int = 500,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    """Bulk-index all documents into the given Elasticsearch index.

    Processes documents in independent chunks. One chunk failing does not
    prevent subsequent chunks from being indexed.

    Retry policy:
      - ConnectionError / ConnectionTimeout: retry with exponential backoff
      - TransportError with status 429/503/504: retry with exponential backoff
      - All other exceptions: log chunk failure, skip chunk, continue

    Args:
        client: An active Elasticsearch client.
        index_name: Name of the target index.
        documents: Mapping doc_id -> Document (from ingestion.loaders).
        batch_size: Number of documents per chunk / bulk request.
        max_retries: Max retry attempts per chunk on transient failures.
        backoff_seconds: Base delay for exponential backoff.

    Returns:
        Stats dict: {
            "documents_loaded": int,
            "documents_indexed": int,
            "failed_documents": int,
            "retries": int,
            "failed_chunks": int,
        }
    """
    actions = [
        _doc_to_action(index_name, doc)
        for doc in documents.values()
    ]
    total_actions = len(actions)
    total_chunks = (total_actions + batch_size - 1) // batch_size

    logger.info(
        "Starting bulk index: %d documents, %d chunks of size %d.",
        total_actions,
        total_chunks,
        batch_size,
    )

    success_count = 0
    error_list: list[dict] = []
    retries = 0
    failed_chunks = 0

    for chunk_idx in range(total_chunks):
        start = chunk_idx * batch_size
        end = start + batch_size
        chunk = actions[start:end]

        logger.info(
            "Chunk %d/%d (%d docs) ...",
            chunk_idx + 1,
            total_chunks,
            len(chunk),
        )

        for attempt in range(max_retries + 1):
            try:
                chunk_success, chunk_errors = bulk(
                    client,
                    chunk,
                    chunk_size=batch_size,
                    raise_on_error=False,
                )
                success_count += chunk_success
                error_list.extend(chunk_errors)
                break  # Chunk succeeded — move to next chunk

            except Exception as exc:
                if _is_retryable(exc) and attempt < max_retries:
                    wait = backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "Chunk %d failed (%s). Retrying in %.1fs ...",
                        chunk_idx + 1,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    retries += 1
                else:
                    # Either non-retryable, or retryable but exhausted retries
                    if _is_retryable(exc):
                        # Exhausted retries on a transient error — systemic issue
                        logger.error(
                            "Chunk %d failed after %d retries: %s",
                            chunk_idx + 1,
                            max_retries,
                            exc,
                        )
                        raise

                    # Non-retryable error — log, mark all docs in chunk failed, continue
                    logger.error(
                        "Chunk %d failed with non-retryable error: %s",
                        chunk_idx + 1,
                        exc,
                    )
                    failed_chunks += 1
                    for action in chunk:
                        error_list.append({
                            "index": {
                                "_index": index_name,
                                "_id": action["_id"],
                                "error": {
                                    "type": type(exc).__name__,
                                    "reason": str(exc),
                                },
                            }
                        })
                    break  # Move to next chunk

    # Log individual document errors (first 5 only, to avoid log spam)
    per_doc_failures = [e for e in error_list if "error" in e.get("index", {})]
    if per_doc_failures:
        logger.warning(
            "%d individual document(s) failed indexing:",
            len(per_doc_failures),
        )
        for err in per_doc_failures[:5]:
            logger.warning("  %s", err)
        if len(per_doc_failures) > 5:
            logger.warning("  ... and %d more.", len(per_doc_failures) - 5)

    if failed_chunks > 0:
        logger.warning("%d chunk(s) failed entirely.", failed_chunks)

    return {
        "documents_loaded": total_actions,
        "documents_indexed": success_count,
        "failed_documents": len(error_list),
        "retries": retries,
        "failed_chunks": failed_chunks,
    }
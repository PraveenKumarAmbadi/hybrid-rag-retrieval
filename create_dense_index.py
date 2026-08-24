#!/usr/bin/env python3
"""Dense vector index creation pipeline.

Loads the SciFact corpus, encodes each document's full_text into dense vectors
using sentence-transformers, builds a FAISS HNSW index, and saves it to disk.
"""

import logging
import time
from pathlib import Path

from configs.loader import load_config
from indexing.dense_encoder import DenseEncoder
from indexing.faiss_indexer import FaissIndex
from ingestion.loaders import load_corpus

logger = logging.getLogger(__name__)


def create_dense_index(recreate: bool = False) -> dict:
    """Build and save a FAISS HNSW dense index from the SciFact corpus.

    Args:
        recreate: If True, overwrite existing index files. If False, skip
                  if index already exists.

    Returns:
        Stats dict with build metrics.
    """
    config = load_config()
    dense_cfg = config["dense"]

    corpus_path = Path(config["data"]["corpus_path"])
    index_path = Path(dense_cfg["index_path"])
    ids_path = Path(dense_cfg["ids_path"])

    # Skip if exists and not recreating
    if index_path.exists() and ids_path.exists() and not recreate:
        logger.info("Dense index already exists at %s. Skipping.", index_path)
        return {"status": "skipped", "index_path": str(index_path)}

    # Phase 1: Load corpus
    logger.info("Loading corpus from %s ...", corpus_path)
    documents = load_corpus(corpus_path)
    logger.info("Loaded %d documents.", len(documents))

    # Phase 2: Encode documents
    logger.info("Encoding %d documents with dense encoder ...", len(documents))
    encoder = DenseEncoder()

    doc_list = list(documents.values())
    texts = [doc.full_text for doc in doc_list]

    start_encode = time.time()
    vectors = encoder.encode_texts(texts, show_progress_bar=True)
    encode_time = time.time() - start_encode
    logger.info(
        "Encoding complete: %d vectors in %.1fs (%.1f docs/sec).",
        len(vectors),
        encode_time,
        len(vectors) / encode_time,
    )

    # Phase 3: Build FAISS HNSW index
    doc_ids = [doc.doc_id for doc in doc_list]
    M = dense_cfg.get("hnsw", {}).get("M", 16)
    ef_construction = dense_cfg.get("hnsw", {}).get("ef_construction", 200)

    start_build = time.time()
    faiss_index = FaissIndex.build(
        vectors=vectors,
        doc_ids=doc_ids,
        M=M,
        ef_construction=ef_construction,
    )
    build_time = time.time() - start_build
    logger.info("Index built in %.1fs.", build_time)

    # Phase 4: Save
    faiss_index.save(index_path, ids_path)

    return {
        "documents_loaded": len(documents),
        "documents_encoded": len(vectors),
        "vector_dimension": vectors.shape[1],
        "encode_time_seconds": round(encode_time, 2),
        "build_time_seconds": round(build_time, 2),
        "index_path": str(index_path),
        "ids_path": str(ids_path),
        "status": "created",
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = create_dense_index()

    print("\n" + "=" * 50)
    print("DENSE INDEX CREATION SUMMARY")
    print("=" * 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    main()

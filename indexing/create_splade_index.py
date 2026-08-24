#!/usr/bin/env python3
"""Create the SciFact SPLADE sparse-vector index in Elasticsearch."""

import logging

from indexing.es_client import get_es_client
from indexing.indexer import create_index  
from indexing.mapping import SCIFACT_SPLADE_MAPPING
from configs.loader import load_config

logger = logging.getLogger(__name__)


def create_splade_index(recreate: bool = False) -> None:
    """Create the scifact_splade index with the SPLADE sparse-vector mapping.

    Args:
        recreate: If True, delete and recreate the index if it exists.
    """
    config = load_config()
    index_name = config["splade"]["index_name"]

    client = get_es_client()

    exists = client.indices.exists(index=index_name)

    if exists and not recreate:
        logger.info("Index '%s' already exists. Skipping creation.", index_name)
        return

    if exists and recreate:
        logger.info("Index '%s' exists — deleting (recreate=True).", index_name)
        client.indices.delete(index=index_name)

    client.indices.create(index=index_name, mappings=SCIFACT_SPLADE_MAPPING)
    logger.info("Index '%s' created with SPLADE sparse_vector mapping.", index_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    create_splade_index()

"""Persistence layer for RAG index data

This module provides functions for saving and loading RAG index data
(FAISS indexes, document corpus, embeddings cache, and metadata) to/from disk.
"""

from .index_persistence import (
    save_index_data,
    load_index_data,
    delete_index_data,
    get_index_metadata
)

__all__ = [
    "save_index_data",
    "load_index_data",
    "delete_index_data",
    "get_index_metadata"
]

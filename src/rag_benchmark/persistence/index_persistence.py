"""Index persistence module for RAG systems

Handles saving and loading of FAISS indexes, document corpus, embeddings cache,
and metadata to/from disk.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def save_index_data(rag_name: str, rag, indices_dir: Path) -> None:
    """Save FAISS index, corpus, embeddings, and metadata to disk
    
    Args:
        rag_name: Name of the RAG instance
        rag: BaselineRAG instance with indexed data
        indices_dir: Base directory for storing index data
        
    Raises:
        RuntimeError: If persistence fails
    """
    try:
        # Create directory for this RAG instance
        rag_dir = indices_dir / rag_name
        rag_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if RAG has indexed data
        if rag.index is None or not rag.documents:
            logger.warning(f"RAG '{rag_name}' has no indexed data to save")
            return
        
        # Import faiss for index saving
        try:
            import faiss
        except ImportError:
            raise RuntimeError("FAISS not installed, cannot save index")
        
        # Define file paths
        index_path = rag_dir / "index.faiss"
        corpus_path = rag_dir / "corpus.json"
        embeddings_path = rag_dir / "embeddings.npy"
        metadata_path = rag_dir / "metadata.json"
        
        # Save FAISS index
        logger.info(f"Saving FAISS index for '{rag_name}' to {index_path}")
        faiss.write_index(rag.index, str(index_path))
        
        # Save document corpus
        logger.info(f"Saving corpus for '{rag_name}' to {corpus_path}")
        corpus_data = {
            "documents": rag.documents,
            "count": len(rag.documents)
        }
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        # Save embeddings cache
        if rag.embeddings_cache is not None:
            logger.info(f"Saving embeddings for '{rag_name}' to {embeddings_path}")
            np.save(str(embeddings_path), rag.embeddings_cache)
        
        # Create and save metadata
        timestamp = datetime.now().isoformat()
        embedding_dim = rag.embeddings_cache.shape[1] if rag.embeddings_cache is not None else 0
        
        metadata = {
            "rag_name": rag_name,
            "document_count": len(rag.documents),
            "embedding_dimension": embedding_dim,
            "index_type": "IndexFlatL2",
            "created_at": timestamp,
            "updated_at": timestamp,
            "file_sizes": {
                "index_faiss": index_path.stat().st_size if index_path.exists() else 0,
                "corpus_json": corpus_path.stat().st_size if corpus_path.exists() else 0,
                "embeddings_npy": embeddings_path.stat().st_size if embeddings_path.exists() else 0
            }
        }
        
        logger.info(f"Saving metadata for '{rag_name}' to {metadata_path}")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully saved index data for '{rag_name}'")
        
    except OSError as e:
        error_msg = f"Failed to save index for RAG '{rag_name}': {type(e).__name__}"
        logger.error(
            error_msg,
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "save_index",
                "error_type": type(e).__name__
            }
        )
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to save index for RAG '{rag_name}': {type(e).__name__}"
        logger.error(
            error_msg,
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "save_index",
                "error_type": type(e).__name__
            }
        )
        raise RuntimeError(error_msg) from e


def load_index_data(rag_name: str, rag, indices_dir: Path) -> bool:
    """Load persisted index data into RAG instance
    
    Args:
        rag_name: Name of the RAG instance
        rag: BaselineRAG instance to load data into
        indices_dir: Base directory for storing index data
        
    Returns:
        True if loading succeeded, False otherwise
    """
    try:
        # Check if directory exists
        rag_dir = indices_dir / rag_name
        if not rag_dir.exists():
            logger.debug(f"No persisted index found for '{rag_name}'")
            return False
        
        # Define file paths
        index_path = rag_dir / "index.faiss"
        corpus_path = rag_dir / "corpus.json"
        embeddings_path = rag_dir / "embeddings.npy"
        metadata_path = rag_dir / "metadata.json"
        
        # Check if all required files exist
        if not all([index_path.exists(), corpus_path.exists(), embeddings_path.exists()]):
            logger.warning(f"Incomplete index data for '{rag_name}', skipping load")
            return False
        
        # Import faiss for index loading
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed, cannot load index")
            return False
        
        # Load FAISS index
        logger.info(f"Loading FAISS index for '{rag_name}' from {index_path}")
        rag.index = faiss.read_index(str(index_path))
        
        # Load document corpus
        logger.info(f"Loading corpus for '{rag_name}' from {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
            rag.documents = corpus_data["documents"]
        
        # Load embeddings cache
        logger.info(f"Loading embeddings for '{rag_name}' from {embeddings_path}")
        rag.embeddings_cache = np.load(str(embeddings_path))
        
        # Verify dimension consistency if embedding model is set
        if rag.embedding_model is not None and rag.embeddings_cache is not None:
            # Get embedding dimension from a test embedding
            try:
                test_embedding = rag.embedding_model.embed_query("test")
                expected_dim = len(test_embedding)
                actual_dim = rag.embeddings_cache.shape[1]
                
                if expected_dim != actual_dim:
                    logger.warning(
                        f"Dimension mismatch for '{rag_name}': "
                        f"expected {expected_dim}, got {actual_dim}. "
                        f"Index may need re-indexing."
                    )
                    # Clear loaded data
                    rag.index = None
                    rag.documents = []
                    rag.embeddings_cache = None
                    return False
            except Exception as e:
                logger.warning(f"Could not verify embedding dimension: {e}")
        
        logger.info(
            f"Successfully loaded index data for '{rag_name}' "
            f"({len(rag.documents)} documents)"
        )
        return True
        
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to load index for RAG '{rag_name}': JSON parse error",
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "load_index",
                "error_type": "JSONDecodeError",
                "file_path": str(corpus_path) if 'corpus_path' in locals() else "unknown"
            }
        )
        return False
    except OSError as e:
        logger.error(
            f"Failed to load index for RAG '{rag_name}': {type(e).__name__}",
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "load_index",
                "error_type": type(e).__name__
            }
        )
        return False
    except Exception as e:
        logger.error(
            f"Failed to load index for RAG '{rag_name}': {type(e).__name__}",
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "load_index",
                "error_type": type(e).__name__
            }
        )
        return False


def delete_index_data(rag_name: str, indices_dir: Path) -> None:
    """Delete all index files for a RAG instance
    
    Args:
        rag_name: Name of the RAG instance
        indices_dir: Base directory for storing index data
    """
    try:
        rag_dir = indices_dir / rag_name
        
        if not rag_dir.exists():
            logger.debug(f"No index directory found for '{rag_name}'")
            return
        
        # Delete all files in the directory
        for file_path in rag_dir.iterdir():
            if file_path.is_file():
                logger.debug(f"Deleting file: {file_path}")
                file_path.unlink()
        
        # Delete the directory itself
        logger.info(f"Deleting index directory for '{rag_name}': {rag_dir}")
        rag_dir.rmdir()
        
        logger.info(f"Successfully deleted index data for '{rag_name}'")
        
    except OSError as e:
        logger.error(
            f"Failed to delete index for RAG '{rag_name}': {type(e).__name__}",
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "delete_index",
                "error_type": type(e).__name__
            }
        )
        # Don't raise, just log - deletion failures shouldn't block operations
    except Exception as e:
        logger.error(
            f"Failed to delete index for RAG '{rag_name}': {type(e).__name__}",
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "delete_index",
                "error_type": type(e).__name__
            }
        )


def get_index_metadata(rag_name: str, indices_dir: Path) -> Optional[Dict]:
    """Get metadata about persisted index without loading full index
    
    Args:
        rag_name: Name of the RAG instance
        indices_dir: Base directory for storing index data
        
    Returns:
        Metadata dictionary if exists, None otherwise
    """
    try:
        rag_dir = indices_dir / rag_name
        metadata_path = rag_dir / "metadata.json"
        
        if not metadata_path.exists():
            logger.debug(f"No metadata found for '{rag_name}'")
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.debug(f"Retrieved metadata for '{rag_name}'")
        return metadata
        
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to read metadata for RAG '{rag_name}': JSON parse error",
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "get_metadata",
                "error_type": "JSONDecodeError",
                "file_path": str(metadata_path) if 'metadata_path' in locals() else "unknown"
            }
        )
        return None
    except OSError as e:
        logger.error(
            f"Failed to read metadata for RAG '{rag_name}': {type(e).__name__}",
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "get_metadata",
                "error_type": type(e).__name__
            }
        )
        return None
    except Exception as e:
        logger.error(
            f"Failed to read metadata for RAG '{rag_name}': {type(e).__name__}",
            exc_info=True,
            extra={
                "rag_name": rag_name,
                "operation": "get_metadata",
                "error_type": type(e).__name__
            }
        )
        return None

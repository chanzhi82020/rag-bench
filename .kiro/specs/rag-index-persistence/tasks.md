# Implementation Plan

- [x] 1. Create persistence layer module
  - Create `src/rag_benchmark/persistence/` directory
  - Create `__init__.py` to export persistence functions
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 1.1 Implement core persistence functions
  - Implement `save_index_data()` function to save FAISS index, corpus, embeddings, and metadata
  - Implement `load_index_data()` function to load persisted data into RAG instance
  - Implement `delete_index_data()` function to remove all index files
  - Implement `get_index_metadata()` function to read metadata without loading full index
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ]* 1.2 Write property test for persistence round-trip
  - **Property 1: Index persistence round-trip**
  - **Validates: Requirements 1.1, 1.2, 1.3, 1.5**

- [x] 1.3 Add error handling to persistence functions
  - Add try-catch blocks for disk I/O errors
  - Add logging for all error scenarios
  - Ensure graceful degradation when persistence fails
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ]* 1.4 Write property test for persistence failure resilience
  - **Property 4: Persistence failure resilience**
  - **Validates: Requirements 4.2**

- [x] 2. Extend BaselineRAG class with persistence methods
  - Add `save_to_disk()` method to save index data
  - Add `load_from_disk()` method to load index data
  - Add `has_index()` method to check if instance has indexed documents
  - Add `get_index_stats()` method to return index statistics
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 2.1 Write property test for index status accuracy
  - **Property 6: Index status accuracy**
  - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

- [x] 3. Update API main.py to integrate persistence
  - Create `INDICES_DIR` constant and ensure directory exists
  - Update `index_documents()` endpoint to call persistence after indexing
  - Update `load_rag_registry()` to load persisted index data
  - Update `delete_rag()` endpoint to delete index files
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ]* 3.1 Write property test for file existence after indexing
  - **Property 2: Index file existence after indexing**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

- [ ]* 3.2 Write property test for complete cleanup on deletion
  - **Property 3: Complete cleanup on deletion**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

- [x] 4. Add index status to API responses
  - Create `IndexStatus` Pydantic model
  - Update `list_rags()` endpoint to include index status for each RAG
  - Add `get_rag_index_status()` endpoint for detailed index information
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 4.1 Write unit tests for index status endpoints
  - Test `list_rags()` includes correct index status
  - Test `get_rag_index_status()` returns accurate information
  - Test status for both indexed and non-indexed RAG instances
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5. Implement re-indexing functionality
  - Update `index_documents()` to detect existing index
  - Implement logic to replace old index files with new ones
  - Update metadata timestamp on re-indexing
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ]* 5.1 Write property test for re-indexing replacement
  - **Property 7: Re-indexing replacement**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

- [x] 6. Add batch loading resilience
  - Update `load_rag_registry()` to handle corrupted index files gracefully
  - Ensure one failed load doesn't prevent other RAG instances from loading
  - Add detailed error logging for failed loads
  - _Requirements: 4.3, 4.4_

- [ ]* 6.1 Write property test for batch loading resilience
  - **Property 5: Batch loading resilience**
  - **Validates: Requirements 4.4**

- [x] 7. Add multi-instance isolation validation
  - Verify each RAG instance uses its own directory
  - Add path validation to prevent directory traversal
  - _Requirements: 2.4_

- [ ]* 7.1 Write property test for multi-instance isolation
  - **Property 8: Multi-instance isolation**
  - **Validates: Requirements 2.4**

- [x] 8. Update frontend to display index status
  - Update RAGPanel component to show index status information
  - Display document count, embedding dimension, file size, and timestamps
  - Add visual indicators for indexed vs non-indexed RAG instances
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

# Design Document: RAG Index Persistence

## Overview

This design implements persistent storage for RAG instance index data, including FAISS vector indexes, document corpus, and embeddings cache. The current system only persists RAG configuration (model IDs, parameters), requiring users to re-index documents after every backend restart. This enhancement enables automatic restoration of indexed data, significantly improving user experience and system efficiency.

The design follows a file-based persistence approach using standard formats (.faiss, .json, .npy) organized in a clear directory structure. It integrates seamlessly with the existing RAG instance lifecycle (create, index, query, delete) and provides robust error handling.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│                                                              │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  RAG Instance  │────────▶│  Persistence     │           │
│  │  Manager       │         │  Layer           │           │
│  └────────────────┘         └──────────────────┘           │
│         │                            │                      │
│         │                            │                      │
│         ▼                            ▼                      │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  BaselineRAG   │         │  File System     │           │
│  │  (in-memory)   │         │  Storage         │           │
│  └────────────────┘         └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  data/indices/   │
                            │  ├── rag1/       │
                            │  │   ├── index.faiss
                            │  │   ├── corpus.json
                            │  │   ├── embeddings.npy
                            │  │   └── metadata.json
                            │  └── rag2/       │
                            └──────────────────┘
```

### Component Interaction Flow

**Indexing Flow:**
1. User calls `/rag/index` API with documents
2. BaselineRAG generates embeddings and builds FAISS index
3. Persistence Layer saves index, corpus, embeddings, and metadata to disk
4. API returns success response

**Loading Flow:**
1. Backend starts and calls `load_rag_registry()`
2. For each RAG config file, load configuration
3. Persistence Layer checks for index data
4. If index data exists, restore FAISS index, corpus, and embeddings
5. RAG instance is ready for queries

**Deletion Flow:**
1. User calls `/rag/{rag_name}` DELETE API
2. Remove RAG instance from memory
3. Persistence Layer deletes all index files and directory
4. API returns success response

## Components and Interfaces

### 1. Persistence Layer Module

**Location:** `src/rag_benchmark/persistence/index_persistence.py`

**Purpose:** Handles all file I/O operations for index data

**Key Functions:**

```python
def save_index_data(rag_name: str, rag: BaselineRAG, indices_dir: Path) -> None:
    """Save FAISS index, corpus, embeddings, and metadata to disk"""
    
def load_index_data(rag_name: str, rag: BaselineRAG, indices_dir: Path) -> bool:
    """Load persisted index data into RAG instance, returns success status"""
    
def delete_index_data(rag_name: str, indices_dir: Path) -> None:
    """Delete all index files for a RAG instance"""
    
def get_index_metadata(rag_name: str, indices_dir: Path) -> Optional[Dict]:
    """Get metadata about persisted index"""
```

### 2. BaselineRAG Extensions

**Location:** `src/rag_benchmark/prepare/baseline_rag.py`

**New Methods:**

```python
def save_to_disk(self, save_path: Path) -> None:
    """Save index data to specified path"""
    
def load_from_disk(self, load_path: Path) -> bool:
    """Load index data from specified path, returns success status"""
    
def has_index(self) -> bool:
    """Check if RAG instance has indexed documents"""
    
def get_index_stats(self) -> Dict:
    """Get statistics about current index"""
```

### 3. API Endpoints

**Modified Endpoints:**

- `POST /rag/index` - After indexing, persist data to disk
- `GET /rag/list` - Include index status in response
- `DELETE /rag/{rag_name}` - Delete index data along with config

**New Endpoints:**

- `GET /rag/{rag_name}/index/status` - Get detailed index information
- `POST /rag/{rag_name}/index/reload` - Reload index from disk

## Data Models

### Directory Structure

```
data/
├── rags/              # RAG configurations (existing)
│   ├── baseline.json
│   └── my_rag.json
└── indices/           # Index data (new)
    ├── baseline/
    │   ├── index.faiss       # FAISS binary index
    │   ├── corpus.json       # Document texts
    │   ├── embeddings.npy    # Numpy array of embeddings
    │   └── metadata.json     # Index metadata
    └── my_rag/
        ├── index.faiss
        ├── corpus.json
        ├── embeddings.npy
        └── metadata.json
```

### File Formats

**1. index.faiss**
- Format: FAISS binary format
- Content: Serialized FAISS IndexFlatL2
- Created by: `faiss.write_index()`
- Loaded by: `faiss.read_index()`

**2. corpus.json**
```json
{
  "documents": [
    "document text 1",
    "document text 2",
    "..."
  ],
  "count": 1000
}
```

**3. embeddings.npy**
- Format: NumPy binary format (.npy)
- Content: 2D array of shape (num_docs, embedding_dim)
- Created by: `np.save()`
- Loaded by: `np.load()`

**4. metadata.json**
```json
{
  "rag_name": "baseline",
  "document_count": 1000,
  "embedding_dimension": 1536,
  "index_type": "IndexFlatL2",
  "created_at": "2025-11-24T10:30:00.000000",
  "updated_at": "2025-11-24T10:30:00.000000",
  "file_sizes": {
    "index_faiss": 6144000,
    "corpus_json": 524288,
    "embeddings_npy": 6144128
  },
  "dataset_info": {
    "dataset_name": "hotpotqa",
    "subset": null
  }
}
```

### API Response Models

**IndexStatus (new Pydantic model):**
```python
class IndexStatus(BaseModel):
    has_index: bool
    document_count: Optional[int] = None
    embedding_dimension: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    total_size_bytes: Optional[int] = None
```

**Updated RAGInfo response:**
```python
{
  "name": "baseline",
  "rag_type": "baseline",
  "model_info": {...},
  "rag_config": {...},
  "index_status": {
    "has_index": true,
    "document_count": 1000,
    "embedding_dimension": 1536,
    "created_at": "2025-11-24T10:30:00",
    "updated_at": "2025-11-24T10:30:00",
    "total_size_bytes": 12812416
  }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Index persistence round-trip

*For any* RAG instance with indexed documents, saving the index data to disk and then loading it back should restore the FAISS index, document corpus, and embeddings cache to equivalent states.

**Validates: Requirements 1.1, 1.2, 1.3, 1.5**

### Property 2: Index file existence after indexing

*For any* RAG instance, after indexing documents, all required files (index.faiss, corpus.json, embeddings.npy, metadata.json) should exist in the dedicated directory.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

### Property 3: Complete cleanup on deletion

*For any* RAG instance with persisted index data, deleting the instance should remove all associated files and the entire index directory.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

### Property 4: Persistence failure resilience

*For any* RAG instance, if index persistence fails, the RAG instance should remain functional in memory for queries.

**Validates: Requirements 4.2**

### Property 5: Batch loading resilience

*For any* set of RAG instances where some have corrupted index data, loading should successfully restore all instances with valid data and skip corrupted ones without crashing.

**Validates: Requirements 4.4**

### Property 6: Index status accuracy

*For any* RAG instance, querying its index status should return accurate information about whether it has indexed documents, the document count, embedding dimension, timestamps, and file sizes.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

### Property 7: Re-indexing replacement

*For any* RAG instance with existing index, re-indexing with new documents should completely replace the old FAISS index, document corpus, embeddings cache, and update the metadata timestamp.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

### Property 8: Multi-instance isolation

*For any* set of RAG instances, each instance's index data should be stored in its own isolated directory, and operations on one instance should not affect others.

**Validates: Requirements 2.4**

## Error Handling

### Error Scenarios and Handling

**1. Disk Write Failures**
- **Scenario:** Insufficient disk space, permission denied, disk I/O error
- **Handling:**
  - Log detailed error with exception traceback
  - Keep RAG instance functional in memory
  - Return error response to user with actionable message
  - Do not crash the application

**2. Disk Read Failures (Loading)**
- **Scenario:** Corrupted files, missing files, incompatible format
- **Handling:**
  - Log error with RAG instance name and file path
  - Skip loading that specific RAG instance
  - Continue loading other RAG instances
  - Mark instance as "not indexed" in memory

**3. FAISS Index Corruption**
- **Scenario:** index.faiss file is corrupted or incompatible
- **Handling:**
  - Catch FAISS exceptions during `faiss.read_index()`
  - Log error and skip index loading
  - RAG instance loads without index (requires re-indexing)

**4. JSON Parse Errors**
- **Scenario:** corpus.json or metadata.json is malformed
- **Handling:**
  - Catch JSON decode exceptions
  - Log error with file path
  - Skip index loading for that RAG instance

**5. NumPy Load Errors**
- **Scenario:** embeddings.npy is corrupted or wrong format
- **Handling:**
  - Catch NumPy exceptions
  - Log error
  - Skip index loading

**6. Dimension Mismatch**
- **Scenario:** Loaded embeddings dimension doesn't match current embedding model
- **Handling:**
  - Detect mismatch during loading
  - Log warning
  - Skip index loading (requires re-indexing with current model)

### Error Logging Format

```python
logger.error(
    f"Failed to save index for RAG '{rag_name}': {error_type}",
    exc_info=True,
    extra={
        "rag_name": rag_name,
        "operation": "save_index",
        "error_type": type(e).__name__,
        "file_path": str(file_path)
    }
)
```

### User-Facing Error Messages

```python
# Disk space error
{
  "error": "Insufficient disk space to save index data",
  "details": "Required: 100MB, Available: 50MB",
  "action": "Free up disk space and try again"
}

# Permission error
{
  "error": "Permission denied when saving index data",
  "details": "Cannot write to /data/indices/my_rag/",
  "action": "Check file permissions for the data directory"
}

# Corrupted index on load
{
  "warning": "Could not load persisted index for RAG 'my_rag'",
  "details": "Index file may be corrupted",
  "action": "Re-index documents to create a new index"
}
```

## Testing Strategy

### Unit Tests

**Test Coverage:**

1. **File I/O Operations**
   - Test saving each file type (FAISS, JSON, NPY)
   - Test loading each file type
   - Test file existence checks
   - Test directory creation

2. **BaselineRAG Extensions**
   - Test `save_to_disk()` method
   - Test `load_from_disk()` method
   - Test `has_index()` method
   - Test `get_index_stats()` method

3. **Error Handling**
   - Test behavior when disk write fails
   - Test behavior when loading corrupted files
   - Test behavior when files are missing
   - Test dimension mismatch detection

4. **API Endpoints**
   - Test `/rag/index` persists data
   - Test `/rag/list` includes index status
   - Test `/rag/{rag_name}/index/status` returns correct info
   - Test `/rag/{rag_name}` DELETE removes index files

### Property-Based Tests

**Testing Framework:** Use `hypothesis` library for Python property-based testing

**Test Configuration:** Each property test should run minimum 100 iterations

**Property Tests:**

1. **Round-trip persistence** (Property 1)
   - Generate random document sets
   - Index, save, load, verify equivalence
   
2. **File existence** (Property 2)
   - Generate random RAG instances
   - Index documents
   - Verify all files exist

3. **Complete cleanup** (Property 3)
   - Generate random RAG instances with indexes
   - Delete instances
   - Verify no files remain

4. **Persistence failure resilience** (Property 4)
   - Simulate disk write failures
   - Verify RAG remains functional

5. **Batch loading resilience** (Property 5)
   - Create multiple RAG instances, corrupt some
   - Load all, verify valid ones load successfully

6. **Index status accuracy** (Property 6)
   - Generate random document sets
   - Index and query status
   - Verify all fields are accurate

7. **Re-indexing replacement** (Property 7)
   - Index documents, then re-index with different documents
   - Verify old data is completely replaced

8. **Multi-instance isolation** (Property 8)
   - Create multiple RAG instances
   - Verify each has isolated storage

### Integration Tests

1. **End-to-end workflow**
   - Create RAG → Index → Restart backend → Query
   - Verify results are consistent

2. **Concurrent operations**
   - Multiple RAG instances indexing simultaneously
   - Verify no data corruption

3. **Large dataset handling**
   - Index 10,000+ documents
   - Verify persistence and loading performance

### Test Utilities

```python
# Test fixtures
@pytest.fixture
def temp_indices_dir(tmp_path):
    """Provide temporary directory for index storage"""
    return tmp_path / "indices"

@pytest.fixture
def sample_rag_instance():
    """Provide a BaselineRAG instance for testing"""
    # Create with mock models
    
@pytest.fixture
def sample_documents():
    """Provide sample document corpus"""
    return ["doc1", "doc2", "doc3"]
```

## Implementation Notes

### Performance Considerations

1. **Batch Operations:** Use batch embedding generation to minimize API calls
2. **Lazy Loading:** Only load index data when RAG instance is first accessed
3. **Async I/O:** Consider using async file operations for large indexes
4. **Compression:** Consider compressing corpus.json for large document sets

### Security Considerations

1. **Path Traversal:** Validate RAG names to prevent directory traversal attacks
2. **File Permissions:** Set appropriate permissions on index directories (0755)
3. **Disk Quotas:** Consider implementing per-RAG disk usage limits

### Scalability Considerations

1. **File System Limits:** Current design suitable for <1000 RAG instances
2. **Large Indexes:** FAISS indexes >1GB may have slow load times
3. **Future Enhancement:** Consider database storage for metadata

### Backward Compatibility

**Note:** This design does NOT maintain backward compatibility with existing RAG instances that have no persisted index data. After deployment:
- Existing RAG instances will load successfully but will have no index
- Users must re-index documents for existing RAG instances
- This is acceptable as the current system already requires re-indexing after restart

## Dependencies

### New Dependencies

- `faiss-cpu` or `faiss-gpu` (already required)
- `numpy` (already required)
- No new external dependencies needed

### Modified Files

1. `src/rag_benchmark/api/main.py`
   - Add index persistence calls
   - Update API responses with index status
   - Add new endpoints

2. `src/rag_benchmark/prepare/baseline_rag.py`
   - Add save/load methods
   - Add index status methods

3. New file: `src/rag_benchmark/persistence/index_persistence.py`
   - Implement all persistence functions

4. New file: `src/rag_benchmark/persistence/__init__.py`
   - Export persistence functions

## Deployment Considerations

### Migration Steps

1. Deploy new code
2. Restart backend
3. Existing RAG instances load without indexes
4. Users re-index documents as needed
5. Future restarts will preserve indexes

### Rollback Plan

If issues arise:
1. Revert to previous code version
2. Index data files remain on disk (harmless)
3. System functions as before (no indexes persisted)

### Monitoring

Add metrics for:
- Index save success/failure rate
- Index load success/failure rate
- Average index file sizes
- Disk space usage for indices directory

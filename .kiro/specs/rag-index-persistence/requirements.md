# Requirements Document

## Introduction

当前RAG系统只持久化配置信息（模型ID、参数等），但不持久化索引后的向量数据和文档内容。这导致每次后端重启后，虽然RAG实例配置被恢复，但向量索引丢失，用户需要重新索引所有文档。本需求旨在实现RAG索引数据的持久化，使得重启后可以直接使用已索引的数据，无需重新索引。

## Glossary

- **RAG System**: 检索增强生成系统，结合向量检索和大语言模型生成的系统
- **Vector Index**: 向量索引，存储文档embeddings的FAISS索引结构
- **Document Corpus**: 文档语料库，被索引的原始文档文本列表
- **Embeddings Cache**: 嵌入向量缓存，文档的向量表示数组
- **Persistence Layer**: 持久化层，负责将内存数据保存到磁盘并恢复的组件
- **FAISS Index**: Facebook AI Similarity Search索引，用于高效向量检索的数据结构
- **RAG Instance**: RAG实例，包含配置、模型和索引数据的完整RAG对象

## Requirements

### Requirement 1

**User Story:** 作为系统管理员，我希望RAG索引数据能够持久化到磁盘，这样后端重启后无需重新索引文档

#### Acceptance Criteria

1. WHEN a user indexes documents to a RAG instance THEN the system SHALL persist the FAISS index to disk
2. WHEN a user indexes documents to a RAG instance THEN the system SHALL persist the document corpus to disk
3. WHEN a user indexes documents to a RAG instance THEN the system SHALL persist the embeddings cache to disk
4. WHEN the backend restarts THEN the system SHALL load persisted index data for all RAG instances
5. WHEN a RAG instance with persisted index is loaded THEN the system SHALL restore the FAISS index, document corpus, and embeddings cache

### Requirement 2

**User Story:** 作为开发者，我希望索引数据的存储格式清晰且可维护，便于调试和数据管理

#### Acceptance Criteria

1. WHEN the system persists index data THEN the system SHALL store FAISS index in binary format with .faiss extension
2. WHEN the system persists index data THEN the system SHALL store document corpus in JSON format with .json extension
3. WHEN the system persists index data THEN the system SHALL store embeddings cache in numpy binary format with .npy extension
4. WHEN the system persists index data THEN the system SHALL organize files in a dedicated directory per RAG instance
5. WHEN the system persists index data THEN the system SHALL include metadata file with index statistics and timestamps

### Requirement 3

**User Story:** 作为用户，我希望删除RAG实例时相关的索引数据也被清理，避免磁盘空间浪费

#### Acceptance Criteria

1. WHEN a user deletes a RAG instance THEN the system SHALL remove the associated FAISS index file
2. WHEN a user deletes a RAG instance THEN the system SHALL remove the associated document corpus file
3. WHEN a user deletes a RAG instance THEN the system SHALL remove the associated embeddings cache file
4. WHEN a user deletes a RAG instance THEN the system SHALL remove the associated metadata file
5. WHEN a user deletes a RAG instance THEN the system SHALL remove the entire index directory for that instance

### Requirement 4

**User Story:** 作为系统管理员，我希望索引持久化操作具有错误处理机制，确保系统稳定性

#### Acceptance Criteria

1. WHEN index persistence fails THEN the system SHALL log the error with detailed information
2. WHEN index persistence fails THEN the system SHALL keep the RAG instance functional in memory
3. WHEN loading persisted index fails THEN the system SHALL log the error and skip that RAG instance
4. WHEN loading persisted index fails THEN the system SHALL continue loading other RAG instances
5. WHEN disk space is insufficient THEN the system SHALL return an error message to the user

### Requirement 5

**User Story:** 作为用户，我希望能够查看RAG实例的索引状态，了解是否已索引以及索引的文档数量

#### Acceptance Criteria

1. WHEN a user queries RAG instance information THEN the system SHALL return whether the instance has indexed documents
2. WHEN a user queries RAG instance information THEN the system SHALL return the number of indexed documents
3. WHEN a user queries RAG instance information THEN the system SHALL return the index creation timestamp
4. WHEN a user queries RAG instance information THEN the system SHALL return the index file size
5. WHEN a user queries RAG instance information THEN the system SHALL return the embedding dimension

### Requirement 6

**User Story:** 作为用户，我希望能够重新索引已有索引的RAG实例，更新索引数据

#### Acceptance Criteria

1. WHEN a user re-indexes a RAG instance with existing index THEN the system SHALL replace the old FAISS index
2. WHEN a user re-indexes a RAG instance with existing index THEN the system SHALL replace the old document corpus
3. WHEN a user re-indexes a RAG instance with existing index THEN the system SHALL replace the old embeddings cache
4. WHEN a user re-indexes a RAG instance with existing index THEN the system SHALL update the metadata timestamp
5. WHEN re-indexing completes THEN the system SHALL persist the new index data to disk

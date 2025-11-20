ents
        return RetrievalResult(contexts=[self.documents[i] for i in indices[0]])
    
    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        # Build prompt
        prompt = f"Context: {contexts}\n\nQuestion: {query}\n\nAnswer:"
        # Call LLM
        response = self.llm.invoke(prompt)
        return GenerationResult(response=response.content)
```

**Batch Optimization**:
```python
def batch_retrieve(self, queries: List[str], top_k: int) -> List[RetrievalResult]:
    # Batch embed all queries at once
    query_embeddings = self.embedding_model.embed_documents(queries)
    # Batch search FAISS
    distances, indices = self.index.search(query_embeddings, top_k)
    # Return results
    return [RetrievalResult(...) for i in range(len(queries))]

def batch_generate(self, queries: List[str], contexts_list: List[List[str]]) 
    -> List[GenerationResult]:
    # Build batch prompts
    prompts = [build_prompt(q, c) for q, c in zip(queries, contexts_list)]
    # Batch call LLM
    responses = self.llm.batch(prompts)
    return [GenerationResult(response=r.content) for r in responses]
```

**Design Rationale**:
- DummyRAG enables testing without infrastructure
- SimpleRAG demonstrates minimal implementation
- BaselineRAG provides production-ready reference
- Batch methods significantly improve throughput

## Data Flow

### Single Record Processing
```
GoldenRecord
    ↓
query = record.user_input
    ↓
retrieval_result = rag.retrieve(query, top_k)
    ↓
generation_result = rag.generate(query, retrieval_result.contexts)
    ↓
SingleTurnSample(
    user_input=query,
    reference=record.reference,
    reference_contexts=record.reference_contexts,
    retrieved_contexts=retrieval_result.contexts,
    response=generation_result.response,
    retrieved_context_ids=retrieval_result.context_ids,
    multi_responses=generation_result.multi_responses
)
```

### Batch Processing
```
[GoldenRecord] → [queries]
    ↓
[RetrievalResult] = rag.batch_retrieve(queries, top_k)
    ↓
[GenerationResult] = rag.batch_generate(queries, [r.contexts for r in retrieval_results])
    ↓
[SingleTurnSample]
```

## Design Principles

1. **Abstraction**:
   - RAGInterface abstracts RAG system details
   - Enables easy integration of any RAG system
   - Supports both custom and framework-based RAGs

2. **Flexibility**:
   - Optional batch methods for performance
   - Configurable error handling
   - Extensible metadata fields

3. **RAGAS Integration**:
   - Direct use of RAGAS data structures
   - No format conversion overhead
   - Seamless pipeline to evaluation

4. **Performance**:
   - Batch processing support
   - Streaming dataset processing
   - Efficient FAISS indexing

5. **Robustness**:
   - Error recovery with skip_on_error
   - Batch fallback to single processing
   - Comprehensive error logging

## Integration Patterns

### Pattern 1: Custom RAG Integration
```python
class MyRAG(RAGInterface):
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.retriever = MyRetriever()
        self.generator = MyGenerator()
    
    def retrieve(self, query: str, top_k: Optional[int]) -> RetrievalResult:
        results = self.retriever.search(query, top_k or self.config.top_k)
        return RetrievalResult(
            contexts=[r.text for r in results],
            context_ids=[r.id for r in results],
            scores=[r.score for r in results]
        )
    
    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        answer = self.generator.generate(query, contexts)
        return GenerationResult(response=answer)
```

### Pattern 2: LangChain Integration
```python
from langchain.chains import RetrievalQA

class LangChainRAG(RAGInterface):
    def __init__(self, qa_chain: RetrievalQA, config: RAGConfig):
        super().__init__(config)
        self.qa_chain = qa_chain
    
    def retrieve(self, query: str, top_k: Optional[int]) -> RetrievalResult:
        docs = self.qa_chain.retriever.get_relevant_documents(query)
        return RetrievalResult(
            contexts=[doc.page_content for doc in docs[:top_k or self.config.top_k]]
        )
    
    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        answer = self.qa_chain.run(query)
        return GenerationResult(response=answer)
```

### Pattern 3: Batch Optimization
```python
class OptimizedRAG(RAGInterface):
    def batch_retrieve(self, queries: List[str], top_k: Optional[int]) 
        -> List[RetrievalResult]:
        # Use vectorized operations
        embeddings = self.embed_batch(queries)
        results = self.index.batch_search(embeddings, top_k)
        return [RetrievalResult(...) for r in results]
    
    def batch_generate(self, queries: List[str], contexts_list: List[List[str]]) 
        -> List[GenerationResult]:
        # Use LLM batch API
        prompts = [self.build_prompt(q, c) for q, c in zip(queries, contexts_list)]
        responses = self.llm.batch_generate(prompts)
        return [GenerationResult(response=r) for r in responses]
```

## Performance Considerations

### Batch Size Selection
- **Small (1-10)**: Better error isolation, more frequent progress updates
- **Medium (10-50)**: Good balance for most use cases
- **Large (50+)**: Maximum throughput, requires more memory

### Memory Management
- Streaming dataset iteration prevents loading all data
- Batch processing trades memory for speed
- FAISS index can be memory-intensive for large corpora

### Error Handling Strategy
- `skip_on_error=True`: Maximize completion, accept partial results
- `skip_on_error=False`: Fail fast, ensure data quality

## Future Enhancements

1. **Async Support**: Async retrieve/generate for concurrent processing
2. **Caching**: Cache retrieval results for repeated queries
3. **Retry Logic**: Automatic retry with exponential backoff
4. **Streaming Generation**: Support streaming LLM responses
5. **Multi-stage RAG**: Support for re-ranking, query rewriting
6. **Distributed Processing**: Support for distributed RAG systems

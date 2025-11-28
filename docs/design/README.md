# RAG Benchmark Design Documentation

## Overview

This directory contains comprehensive design documentation for the RAG Benchmark framework. Each module is documented separately with detailed architecture, design rationale, and usage patterns.

## Module Structure

```
rag_benchmark/
├── datasets/      # Dataset management and loading
├── prepare/       # Experiment dataset preparation
├── evaluate/      # RAG system evaluation
├── analysis/      # Result comparison and visualization
└── api/          # Web API service
```

## Module Documentation

### [Datasets Module](datasets/overview.md)

**Purpose**: Standardized dataset management for RAG evaluation

**Key Components**:
- GoldenDataset: High-level dataset interface with functional operations
- DatasetRegistry: Centralized dataset management
- Loaders: Pluggable data loading (JSONL, etc.)
- Converters: Transform public datasets to Golden format
- Validators: Format and quality validation

**Design Highlights**:
- Iterator-based lazy loading for memory efficiency
- Functional operations (filter, map, sample)
- Extensible converter architecture
- Comprehensive validation

**Documentation**:
- [datasets/overview.md](datasets/overview.md) - Module architecture and design
- [datasets/converters.md](datasets/converters.md) - Detailed format analysis and conversion logic
- [datasets/dataset-comparison.md](datasets/dataset-comparison.md) - Dataset comparison and quick reference

### [Prepare Module](prepare/overview.md)

**Purpose**: Convert Golden Datasets to Experiment Datasets by invoking RAG systems

**Key Components**:
- RAGInterface: Abstract interface for RAG systems
- prepare_experiment_dataset(): Core preparation function
- BaselineRAG: FAISS-based reference implementation
- RetrievalResult/GenerationResult: Structured outputs

**Design Highlights**:
- Minimal interface (retrieve + generate)
- Batch processing support with fallback
- Direct RAGAS integration
- Robust error handling

**Read More**: [prepare/overview.md](prepare/overview.md)

### [Evaluate Module](evaluate/overview.md)

**Purpose**: Comprehensive RAG system evaluation using RAGAS and traditional metrics

**Key Components**:
- evaluate_e2e/retrieval/generation(): Specialized evaluation functions
- RAGAS metrics: LLM-based semantic evaluation
- Traditional IR metrics: Statistical evaluation (recall@k, NDCG, etc.)
- Metric adapters: Wrap IR metrics as RAGAS metrics

**Design Highlights**:
- Dual metric system (LLM-based + statistical)
- Conditional metric selection based on data
- Custom model configuration
- Cost-aware design

**Read More**: [evaluate/overview.md](evaluate/overview.md)

### [Analysis Module](analysis/overview.md)

**Purpose**: Result comparison and visualization for multi-model analysis

**Key Components**:
- ResultComparison: Multi-model comparison with statistics
- compare_results(): Main comparison function
- Visualization functions: plot_metrics, plot_comparison, plot_distribution
- Error analysis: Identify worst-performing samples

**Design Highlights**:
- Comprehensive statistics (mean, std, min, max)
- Multiple visualization types
- Error case analysis
- Export capabilities

**Read More**: [analysis/overview.md](analysis/overview.md)

### [API Module](api/overview.md)

**Purpose**: FastAPI-based web service for remote RAG evaluation

**Key Components**:
- Model Registry: Centralized model management
- RAG APIs: Create and query RAG systems
- Evaluation APIs: Async evaluation with progress tracking
- Dataset APIs: Dataset exploration and statistics

**Design Highlights**:
- RESTful design
- Async task processing
- Persistent state management
- Model reusability

**Read More**: [api/overview.md](api/overview.md)

## System Architecture

### High-Level Flow

```
┌─────────────┐
│   Datasets  │  Load and validate golden datasets
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Prepare   │  Invoke RAG system to generate experiment data
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Evaluate   │  Compute RAGAS and IR metrics
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Analysis   │  Compare results and visualize
└─────────────┘

       ▲
       │
┌─────────────┐
│     API     │  Web interface for all operations
└─────────────┘
```

### Data Flow

```
Golden Dataset (qac.jsonl + corpus.jsonl)
    ↓
GoldenDataset.load()
    ↓
[GoldenRecord] → prepare_experiment_dataset(rag_system)
    ↓
EvaluationDataset (RAGAS format)
    ↓
evaluate_e2e/retrieval/generation()
    ↓
EvaluationResult (RAGAS format)
    ↓
compare_results([result1, result2, ...])
    ↓
ResultComparison + Visualizations
```

### Key Data Structures

**GoldenRecord** (datasets):
```python
user_input: str                    # Question
reference: str                     # Ground truth answer
reference_contexts: List[str]      # Ground truth contexts
reference_context_ids: List[str]   # Context IDs (optional)
```

**SingleTurnSample** (RAGAS, used in prepare/evaluate):
```python
user_input: str                    # Question
reference: str                     # Ground truth answer
reference_contexts: List[str]      # Ground truth contexts
retrieved_contexts: List[str]      # RAG retrieved contexts
response: str                      # RAG generated answer
retrieved_context_ids: List[str]   # Retrieved context IDs (optional)
```

**EvaluationResult** (RAGAS, output of evaluate):
```python
# Contains DataFrame with:
# - Input fields (user_input, reference, etc.)
# - Metric scores (faithfulness, answer_relevancy, etc.)
```

## Design Principles

### 1. Modularity
- Each module has clear responsibilities
- Minimal coupling between modules
- Pluggable implementations (loaders, converters, RAG systems)

### 2. RAGAS Integration
- Direct use of RAGAS data structures
- No format conversion overhead
- Leverage RAGAS ecosystem

### 3. Extensibility
- Abstract base classes for custom implementations
- Registry patterns for plugins
- Metadata fields for custom attributes

### 4. Performance
- Lazy loading and streaming
- Batch processing support
- Efficient data structures (FAISS, pandas)

### 5. Robustness
- Comprehensive validation
- Error recovery mechanisms
- Detailed error reporting

### 6. Usability
- High-level interfaces for common tasks
- Sensible defaults
- Rich documentation and examples

## Common Patterns

### Pattern 1: End-to-End Evaluation

```python
from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.prepare import prepare_experiment_dataset, BaselineRAG
from rag_benchmark.evaluate import evaluate_e2e
from rag_benchmark.analysis import compare_results, plot_metrics

# 1. Load dataset
dataset = GoldenDataset("hotpotqa", subset="distractor")

# 2. Create RAG system
rag = BaselineRAG(embedding_model=embeddings, llm=llm)

# 3. Prepare experiment dataset
exp_ds = prepare_experiment_dataset(dataset, rag)

# 4. Evaluate
result = evaluate_e2e(exp_ds, experiment_name="my_rag")

# 5. Analyze
print(result.to_pandas().mean())
```

### Pattern 2: Multi-Model Comparison

```python
# Prepare datasets for multiple RAG systems
exp_ds1 = prepare_experiment_dataset(dataset, rag1)
exp_ds2 = prepare_experiment_dataset(dataset, rag2)

# Evaluate both
result1 = evaluate_e2e(exp_ds1, experiment_name="baseline")
result2 = evaluate_e2e(exp_ds2, experiment_name="improved")

# Compare
comparison = compare_results([result1, result2], names=["Baseline", "Improved"])
print(comparison.summary())
plot_metrics(comparison)
```

### Pattern 3: Custom RAG Integration

```python
from rag_benchmark.prepare import RAGInterface, RetrievalResult, GenerationResult

class MyRAG(RAGInterface):
    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        # Your retrieval logic
        return RetrievalResult(contexts=[...], scores=[...])
    
    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        # Your generation logic
        return GenerationResult(response="...")

# Use in pipeline
rag = MyRAG()
exp_ds = prepare_experiment_dataset(dataset, rag)
result = evaluate_e2e(exp_ds)
```

### Pattern 4: Dataset Conversion

```python
from scripts.converters import HotpotQAConverter

# Convert public dataset to Golden format
converter = HotpotQAConverter(
    output_dir="data/hotpotqa",
    variant="distractor"
)
result = converter.convert("hotpotqa/hotpot_qa")

# Load converted dataset
dataset = GoldenDataset("hotpotqa", subset="distractor")
```

### Pattern 5: API-Based Evaluation

```python
import requests

# Register models
requests.post("http://localhost:8000/models/register", json={
    "model_id": "my-llm",
    "model_name": "gpt-3.5-turbo",
    "model_type": "llm",
    "api_key": "sk-..."
})

# Create RAG
requests.post("http://localhost:8000/rag/create", json={
    "name": "my_rag",
    "model_info": {
        "llm_model_id": "my-llm",
        "embedding_model_id": "my-embedding"
    }
})

# Start evaluation
response = requests.post("http://localhost:8000/evaluate/start", json={
    "dataset_name": "hotpotqa",
    "rag_name": "my_rag",
    "eval_type": "e2e"
})
task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/evaluate/status/{task_id}")
```

## Technology Stack

### Core Dependencies
- **Python 3.11+**: Modern Python features
- **Pydantic**: Data validation and settings
- **RAGAS**: LLM-based evaluation metrics
- **LangChain**: LLM and embedding abstractions

### Data Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **FAISS**: Vector similarity search

### Web Framework
- **FastAPI**: Modern async web framework
- **uvicorn**: ASGI server

### Visualization
- **matplotlib**: Plotting and visualization

### Optional
- **datasets**: HuggingFace datasets library
- **tqdm**: Progress bars

## Performance Characteristics

### Memory Usage
- **Datasets**: Streaming support for large datasets
- **Prepare**: Batch processing configurable
- **Evaluate**: Full dataset in memory (RAGAS requirement)
- **Analysis**: Multiple results in memory

### Computation Time
- **Datasets**: O(n) for loading
- **Prepare**: O(n * retrieval_time + n * generation_time)
- **Evaluate**: O(n * m * llm_time) where m = number of LLM metrics
- **Analysis**: O(n * m) for comparison

### Cost Considerations
- **Prepare**: API costs for RAG system
- **Evaluate**: API costs for evaluation LLM
- **Traditional IR metrics**: Free (no LLM)

## Testing Strategy

### Unit Tests
- Schema validation
- Metric computation
- Data transformations

### Integration Tests
- End-to-end pipeline
- API endpoints
- Dataset conversion

### Example Tests
- Example scripts as smoke tests
- Verify basic functionality
- Catch breaking changes

## Future Roadmap

### Short Term
1. Async evaluation support
2. Result caching
3. More dataset converters
4. Enhanced visualizations

### Medium Term
1. Distributed evaluation
2. Custom metric support
3. Multi-language support
4. Advanced analysis tools

### Long Term
1. AutoML for RAG optimization
2. Real-time evaluation
3. Production monitoring
4. Enterprise features

## Contributing

### Adding a New Dataset
1. Create converter in `datasets/converters/`
2. Register in `datasets/registry.py`
3. Add tests and documentation

### Adding a New Metric
1. Implement metric function in `evaluate/metrics_*.py`
2. Create RAGAS adapter class
3. Add to metric groups
4. Document usage

### Adding a New RAG System
1. Implement `RAGInterface`
2. Add example in `examples/`
3. Document integration pattern

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)

## Support

For questions and issues:
- Check module-specific documentation
- Review example scripts
- Open GitHub issue
- Consult API documentation at `/docs`

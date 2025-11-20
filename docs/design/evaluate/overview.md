# Evaluate Module Design

## Overview

The evaluate module is the core evaluation stage of the RAG pipeline, providing comprehensive assessment of RAG systems using both RAGAS metrics (LLM-based) and traditional IR metrics (statistical).

## Architecture

```
evaluate/
├── evaluator.py           # Core evaluation functions
└── metrics_retrieval.py   # Traditional IR metrics
```

## Core Components

### 1. Evaluation Functions

The module provides three specialized evaluation functions:

**evaluate_e2e()**: End-to-end evaluation
```python
def evaluate_e2e(
    dataset: EvaluationDataset,
    experiment_name: Optional[str] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[Union[BaseRagasEmbeddings, LangchainEmbeddings]] = None,
    callbacks: Optional[Callbacks] = None,
    run_config: Optional[RunConfig] = None,
    show_progress: bool = True,
    **kwargs
) -> Union[EvaluationResult, Executor]
```

**Metrics Used**:
- faithfulness (generation quality)
- answer_relevancy (end-to-end quality)
- context_precision (retrieval quality)
- context_recall (retrieval quality)
- recall@k, precision@k, f1@k, ndcg@k (if context IDs available)

**evaluate_retrieval()**: Retrieval-specific evaluation
```python
def evaluate_retrieval(
    dataset: EvaluationDataset,
    experiment_name: Optional[str] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[Union[BaseRagasEmbeddings, LangchainEmbeddings]] = None,
    **kwargs
) -> Union[EvaluationResult, Executor]
```

**Metrics Used**:
- context_recall (RAGAS)
- context_precision (RAGAS)
- recall@k, precision@k, f1@k, ndcg@k (traditional IR)

**evaluate_generation()**: Generation-specific evaluation
```python
def evaluate_generation(
    dataset: EvaluationDataset,
    experiment_name: Optional[str] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[Union[BaseRagasEmbeddings, LangchainEmbeddings]] = None,
    **kwargs
) -> Union[EvaluationResult, Executor]
```

**Metrics Used**:
- faithfulness (answer grounded in context)
- answer_correctness (answer matches reference)

**Design Rationale**:
- Specialized functions enable focused evaluation
- Consistent interface across all evaluation types
- Direct RAGAS integration for seamless workflow

### 2. RAGAS Metrics

RAGAS provides LLM-based metrics that assess semantic quality:

**faithfulness**: Measures if the answer is grounded in the retrieved contexts
- Checks for hallucinations
- Verifies factual consistency
- Range: 0-1 (higher is better)

**answer_relevancy**: Measures if the answer addresses the question
- Semantic similarity between question and answer
- Considers answer completeness
- Range: 0-1 (higher is better)

**answer_correctness**: Measures if the answer matches the reference
- Combines semantic similarity and factual overlap
- Weighted combination of similarity and F1
- Range: 0-1 (higher is better)

**context_recall**: Measures if reference contexts are retrieved
- Checks if ground truth contexts appear in retrieved contexts
- Requires reference_contexts field
- Range: 0-1 (higher is better)

**context_precision**: Measures if retrieved contexts are relevant
- Checks if retrieved contexts are useful for answering
- Uses LLM to judge relevance
- Range: 0-1 (higher is better)

**Design Rationale**:
- LLM-based metrics capture semantic quality
- Complement traditional metrics
- Require evaluation LLM (cost consideration)

### 3. Traditional IR Metrics

Statistical metrics that don't require LLMs:

**recall@k**: Proportion of relevant documents retrieved
```python
def recall_at_k(
    retrieved_ids: List[str],
    reference_ids: List[str],
    k: Optional[int] = None
) -> float:
    """
    Recall@K = |retrieved[:k] ∩ reference| / |reference|
    """
```

**precision@k**: Proportion of retrieved documents that are relevant
```python
def precision_at_k(
    retrieved_ids: List[str],
    reference_ids: List[str],
    k: Optional[int] = None
) -> float:
    """
    Precision@K = |retrieved[:k] ∩ reference| / k
    """
```

**f1@k**: Harmonic mean of precision and recall
```python
def f1_at_k(
    retrieved_ids: List[str],
    reference_ids: List[str],
    k: Optional[int] = None
) -> float:
    """
    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
    """
```

**ndcg@k**: Normalized Discounted Cumulative Gain
```python
def ndcg_at_k(
    retrieved_ids: List[str],
    reference_ids: List[str],
    k: Optional[int] = None
) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    DCG@K = Σ(rel_i / log2(i+1)) for i in 1..k
    """
```

**mrr**: Mean Reciprocal Rank
```python
def mean_reciprocal_rank(
    retrieved_ids: List[str],
    reference_ids: List[str]
) -> float:
    """
    MRR = 1 / rank_of_first_relevant_item
    """
```

**map**: Mean Average Precision
```python
def average_precision(
    retrieved_ids: List[str],
    reference_ids: List[str]
) -> float:
    """
    AP = (Σ(P@k * rel_k)) / |reference|
    """
```

**Design Rationale**:
- No LLM required (fast and free)
- Standard IR metrics for comparison
- Require context IDs (retrieved_context_ids, reference_context_ids)

### 4. RAGAS Metric Integration

Traditional IR metrics are wrapped as RAGAS metrics for seamless integration:

```python
@dataclass
class RecallAtK(SingleTurnMetric):
    k: int = 5
    name: str = field(default=f"recall@{k}")
    
    _required_columns: dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "retrieved_context_ids",
                "reference_context_ids",
            },
        }
    )
    
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_ids = sample.retrieved_context_ids
        reference_ids = sample.reference_context_ids
        return recall_at_k(retrieved_ids, reference_ids, self.k)
```

**Design Pattern**: Adapter pattern
- Wraps functional metrics as RAGAS SingleTurnMetric
- Enables use with RAGAS evaluate() function
- Maintains consistency with RAGAS API

**Available Metric Classes**:
- `RecallAtK(k=5)`
- `PrecisionAtK(k=5)`
- `F1AtK(k=5)`
- `NDCGAtK(k=10)`
- `MRRMetric()`
- `MAPMetric()`

## Evaluation Flow

### Standard Evaluation Flow
```
EvaluationDataset (from prepare module)
    ↓
Select metrics (e2e, retrieval, or generation)
    ↓
evaluate_xxx(dataset, metrics, llm, embeddings)
    ↓
For each sample:
    1. Extract required fields
    2. Compute each metric
    3. Store results
    ↓
EvaluationResult (RAGAS format)
    ↓
result.to_pandas() → DataFrame with metric columns
```

### Metric Computation Flow

**RAGAS Metrics** (e.g., faithfulness):
```
Sample → LLM Prompt → LLM Response → Parse Score → 0-1 value
```

**Traditional Metrics** (e.g., recall@k):
```
Sample → Extract IDs → Set Operations → Compute Ratio → 0-1 value
```

### Conditional Metric Selection

The module automatically includes traditional IR metrics only if context IDs are available:

```python
sample = dataset[0]
if sample.reference_context_ids and sample.retrieved_context_ids:
    # Add traditional IR metrics
    metrics.extend([RecallAtK(), PrecisionAtK(), F1AtK(), NDCGAtK()])
```

**Design Rationale**:
- Graceful degradation when IDs unavailable
- Automatic metric selection based on data
- No manual configuration needed

## Data Requirements

### For RAGAS Metrics

**faithfulness**:
- `response` (generated answer)
- `retrieved_contexts` (contexts used for generation)

**answer_relevancy**:
- `user_input` (question)
- `response` (generated answer)

**answer_correctness**:
- `response` (generated answer)
- `reference` (ground truth answer)

**context_recall**:
- `reference_contexts` (ground truth contexts)
- `retrieved_contexts` (RAG retrieved contexts)

**context_precision**:
- `user_input` (question)
- `retrieved_contexts` (RAG retrieved contexts)
- `reference` (ground truth answer)

### For Traditional IR Metrics

**All IR metrics require**:
- `retrieved_context_ids` (IDs of retrieved documents)
- `reference_context_ids` (IDs of ground truth documents)

**Design Rationale**:
- Clear separation of requirements
- Enables partial evaluation when data incomplete
- Supports different dataset formats

## Model Configuration

### Using Custom Models

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configure evaluation models
eval_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="your-api-key",
    temperature=0.0  # Deterministic for evaluation
)

eval_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="your-api-key"
)

# Use in evaluation
result = evaluate_e2e(
    dataset=exp_ds,
    llm=eval_llm,
    embeddings=eval_embeddings,
    experiment_name="my_evaluation"
)
```

**Design Rationale**:
- Separate evaluation models from RAG models
- Supports any LangChain-compatible model
- Enables cost/quality tradeoffs

### Model Selection Guidelines

**For LLM**:
- GPT-4: Highest quality, most expensive
- GPT-3.5-turbo: Good balance
- Local models: Free but may be less accurate

**For Embeddings**:
- text-embedding-3-small: Fast and cheap
- text-embedding-3-large: Higher quality
- Local embeddings: Free but may affect quality

## Results Format

### EvaluationResult (RAGAS)

```python
result = evaluate_e2e(dataset)

# Convert to DataFrame
df = result.to_pandas()
# Columns: user_input, response, reference, faithfulness, answer_relevancy, ...

# Get mean scores
mean_scores = df.mean()
# faithfulness: 0.85, answer_relevancy: 0.78, ...

# Access individual samples
for idx, row in df.iterrows():
    print(f"Question: {row['user_input']}")
    print(f"Faithfulness: {row['faithfulness']}")
```

### Metric Aggregation

```python
# Mean (default)
mean_faithfulness = df['faithfulness'].mean()

# Median (robust to outliers)
median_faithfulness = df['faithfulness'].median()

# Standard deviation
std_faithfulness = df['faithfulness'].std()

# Percentiles
p95_faithfulness = df['faithfulness'].quantile(0.95)
```

## Design Principles

1. **Modularity**:
   - Separate functions for different evaluation types
   - Independent metric implementations
   - Pluggable model configuration

2. **RAGAS Integration**:
   - Direct use of RAGAS evaluate() function
   - Consistent with RAGAS ecosystem
   - Leverages RAGAS optimizations

3. **Flexibility**:
   - Support both RAGAS and traditional metrics
   - Optional model configuration
   - Conditional metric selection

4. **Performance**:
   - Batch processing in RAGAS
   - Efficient set operations for IR metrics
   - Progress tracking for long evaluations

5. **Cost Awareness**:
   - Traditional metrics are free
   - LLM metrics incur API costs
   - Clear separation enables cost control

## Usage Patterns

### Pattern 1: Quick E2E Evaluation
```python
result = evaluate_e2e(exp_ds, experiment_name="quick_test")
print(result.to_pandas().mean())
```

### Pattern 2: Retrieval-Only Evaluation
```python
result = evaluate_retrieval(exp_ds, experiment_name="retrieval_test")
print(f"Context Recall: {result.to_pandas()['context_recall'].mean():.3f}")
```

### Pattern 3: Custom Model Evaluation
```python
result = evaluate_e2e(
    exp_ds,
    llm=custom_llm,
    embeddings=custom_embeddings,
    experiment_name="custom_model_eval"
)
```

### Pattern 4: Cost-Effective Evaluation
```python
# Use only traditional IR metrics (no LLM cost)
from rag_benchmark.evaluate.metrics_retrieval import compute_retrieval_metrics

retrieved_ids_list = [sample.retrieved_context_ids for sample in exp_ds]
reference_ids_list = [sample.reference_context_ids for sample in exp_ds]

metrics = compute_retrieval_metrics(
    retrieved_ids_list,
    reference_ids_list,
    k_values=[1, 3, 5, 10]
)
print(metrics)
```

## Performance Considerations

### LLM Evaluation Cost
- Each RAGAS metric requires 1-3 LLM calls per sample
- For 100 samples with 3 metrics: ~300-900 LLM calls
- Use cheaper models (gpt-3.5-turbo) for development
- Use better models (gpt-4) for final evaluation

### Evaluation Time
- RAGAS metrics: ~1-5 seconds per sample
- Traditional metrics: <0.01 seconds per sample
- Use smaller datasets for rapid iteration
- Use traditional metrics for quick feedback

### Memory Usage
- EvaluationDataset loaded into memory
- Results stored in DataFrame
- For large datasets, consider batch evaluation

## Future Enhancements

1. **Custom Metrics**: Support user-defined metrics
2. **Async Evaluation**: Parallel metric computation
3. **Caching**: Cache LLM responses for repeated evaluations
4. **Streaming Results**: Stream results for large datasets
5. **Multi-language Support**: Language-specific metrics
6. **Confidence Intervals**: Statistical significance testing
7. **Metric Correlation**: Analyze metric relationships

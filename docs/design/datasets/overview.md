# Datasets Module Design

## Overview

The datasets module provides standardized dataset management for RAG evaluation, including Golden Dataset loading, conversion, validation, and manipulation capabilities.

## Architecture

```
datasets/
├── schemas/          # Data schemas and validation
│   └── golden.py    # GoldenRecord, CorpusRecord, DatasetMetadata
├── loaders/         # Data loading implementations
│   ├── base.py      # BaseLoader abstract interface
│   └── jsonl.py     # JSONL format loader
├── converters/      # Dataset format converters
│   ├── base.py      # BaseConverter abstract interface
│   ├── hotpotqa.py  # HotpotQA converter
│   ├── nq.py        # Natural Questions converter
│   └── xquad.py     # XQuAD converter
├── validators/      # Data validation
│   ├── format.py    # Format validation
│   └── quality.py   # Quality metrics validation
├── golden.py        # High-level GoldenDataset interface
└── registry.py      # Dataset registry and management
```

## Core Components

### 1. Data Schemas

**GoldenRecord**: Represents a single Q&A record with contexts
```python
@dataclass
class GoldenRecord:
    user_input: str                          # User question
    reference: str                           # Reference answer
    reference_contexts: List[str]            # Relevant context passages
    reference_context_ids: Optional[List[str]]  # Context IDs
    metadata: Dict[str, Any]                 # Additional metadata
```

**CorpusRecord**: Represents a document in the corpus
```python
@dataclass
class CorpusRecord:
    reference_context: str      # Document content
    reference_context_id: str   # Unique document ID
    title: str                  # Document title
    metadata: Dict[str, Any]    # Additional metadata
```

**Design Rationale**:
- Separation of Q&A records and corpus enables efficient storage and retrieval
- Optional `reference_context_ids` supports both ID-based and direct text references
- Metadata fields provide extensibility for dataset-specific information

### 2. GoldenDataset Interface

High-level interface for unified dataset access with functional operations:

```python
class GoldenDataset:
    def __iter__(self) -> Iterator[GoldenRecord]
    def filter(self, predicate) -> DatasetView
    def sample(self, n: int, seed: Optional[int]) -> List[GoldenRecord]
    def create_subset(self, n: int, seed: Optional[int]) -> GoldenDataset
    def stats(self) -> Dict[str, Any]
    def validate(self) -> Dict[str, Any]
    def export(self, output_path, format_type="jsonl")
```

**Key Features**:
- **Lazy Loading**: Iterator-based access for memory efficiency
- **Functional Operations**: filter, map, collect pattern inspired by Spark/Pandas
- **DatasetView**: Immutable view with chained transformations
- **Subset Creation**: Returns new GoldenDataset (not just list) for compatibility

**Design Pattern**: Builder + Iterator pattern for flexible data manipulation

### 3. Dataset Registry

Centralized registry for managing available datasets:

```python
class DatasetRegistry:
    def register_dataset(self, dataset_info: Dict[str, Any])
    def list_datasets(self) -> List[Dict[str, Any]]
    def get_loader(self, name: str, subset: Optional[str]) -> BaseLoader
    def is_dataset_available(self, name: str, subset: Optional[str]) -> bool
```

**Built-in Datasets**:
- HotpotQA (distractor, fullwiki)
- Natural Questions
- XQuAD (multilingual)
- Customer Service (private)

**Design Rationale**:
- Centralized management simplifies dataset discovery
- Subset support enables dataset variants (e.g., language-specific)
- Lazy loader instantiation reduces startup overhead

### 4. Loaders

**BaseLoader**: Abstract interface for data loading
```python
class BaseLoader(ABC):
    @abstractmethod
    def load_golden_records(self) -> Iterator[GoldenRecord]
    
    @abstractmethod
    def load_corpus_records(self) -> Iterator[CorpusRecord]
    
    def validate(self) -> ValidationResult
    def count_records(self) -> int
    def get_sample(self, n: int) -> List[GoldenRecord]
```

**JSONLLoader**: JSONL format implementation
- Streaming support for large datasets
- Automatic validation with Pydantic models
- Error recovery (skip invalid lines)

**Design Pattern**: Strategy pattern for pluggable loaders

### 5. Converters

**BaseConverter**: Abstract interface for dataset conversion
```python
class BaseConverter(ABC):
    @abstractmethod
    def load_source_data(self, source_path) -> Iterator[Dict]
    
    @abstractmethod
    def convert_record(self, source_record) -> List[Tuple[GoldenRecord, List[CorpusRecord]]]
    
    @abstractmethod
    def create_metadata(self, source_path, num_records) -> DatasetMetadata
    
    def convert(self, source_path) -> ConversionResult
```

**Converter Implementations**:
- **HotpotQAConverter**: Handles multi-hop reasoning with supporting facts
  - Challenge: Supporting facts resolution (title mismatches, bridge questions)
  - Solution: Multi-level matching (exact, entity-based, content-based)
  - Handles both distractor (10 paragraphs) and fullwiki variants
  
- **NaturalQuestionsConverter**: Extracts long/short answers from HTML
  - Challenge: HTML parsing and token span extraction
  - Solution: HTML cleaning + token reconstruction
  - Extracts relevant passages using long_answer spans
  
- **XQuADConverter**: Processes multilingual SQuAD-format data
  - Challenge: Multiple Q&A per paragraph, corpus deduplication
  - Solution: Global paragraph counter, one GoldenRecord per Q&A
  - Supports multiple languages (zh, en, etc.)

**Key Features**:
- Batch processing for efficiency
- Corpus deduplication (global corpus_map)
- Progress tracking with tqdm
- Error recovery and reporting

**Design Rationale**:
- Template Method pattern for consistent conversion flow
- Batch writing reduces I/O overhead
- Deduplication prevents corpus bloat

**Detailed Format Analysis**: See [converters.md](converters.md) for:
- Source format specifications
- Conversion challenges and solutions
- Step-by-step conversion logic
- Example conversions for each dataset

### 6. Validators

**FormatValidator**: Validates data structure and format
- Field presence and type checking
- Length constraints
- ID uniqueness and consistency
- Cross-reference validation (golden ↔ corpus)

**QualityValidator**: Analyzes dataset quality
- Question quality metrics (length, type, patterns)
- Answer quality metrics (completeness, structure)
- Context quality metrics (coverage, relevance)
- Relevance scoring (Q-A, Q-C overlap)

**Design Rationale**:
- Separation of format vs. quality concerns
- Statistical analysis for quality assessment
- Actionable warnings and errors

## Data Flow

### Loading Flow
```
Registry → Loader → Iterator[GoldenRecord]
                 → Iterator[CorpusRecord]
```

### Conversion Flow
```
Source Dataset → Converter.load_source_data()
              → Converter.convert_record()
              → Batch Writing (qac.jsonl, corpus.jsonl)
              → Metadata Generation
```

### Validation Flow
```
Dataset → FormatValidator → Structure checks
       → QualityValidator → Statistical analysis
       → ValidationResult
```

## Design Principles

1. **Separation of Concerns**:
   - Schemas: Data structure
   - Loaders: Data access
   - Converters: Data transformation
   - Validators: Data quality

2. **Extensibility**:
   - Abstract base classes for custom implementations
   - Registry pattern for plugin architecture
   - Metadata fields for custom attributes

3. **Performance**:
   - Streaming/iterator-based processing
   - Lazy loading
   - Batch operations
   - Corpus deduplication

4. **Robustness**:
   - Comprehensive validation
   - Error recovery
   - Detailed error reporting
   - Type safety with Pydantic

5. **Usability**:
   - High-level GoldenDataset interface
   - Functional operations (filter, map, sample)
   - Automatic format detection
   - Rich statistics and metadata

## File Format

### qac.jsonl (Question-Answer-Context)
```json
{"user_input": "...", "reference": "...", "reference_contexts": [...], "reference_context_ids": [...]}
```

### corpus.jsonl
```json
{"reference_context": "...", "reference_context_id": "...", "title": "..."}
```

### metadata.json
```json
{
  "name": "dataset_name",
  "version": "1.0",
  "description": "...",
  "source": "...",
  "size": 1000,
  "domain": "general",
  "language": "en"
}
```

## Usage Patterns

### Pattern 1: Simple Loading
```python
dataset = GoldenDataset("hotpotqa", subset="distractor")
for record in dataset:
    process(record)
```

### Pattern 2: Filtered Processing
```python
dataset = GoldenDataset("xquad", subset="zh")
short_questions = dataset.filter(lambda r: len(r.user_input) < 100)
for record in short_questions:
    process(record)
```

### Pattern 3: Subset Creation
```python
dataset = GoldenDataset("hotpotqa")
subset = dataset.create_subset(100, seed=42)  # Returns GoldenDataset
# subset can be used anywhere GoldenDataset is expected
```

### Pattern 4: Dataset Conversion
```python
converter = HotpotQAConverter(output_dir="data/hotpotqa", variant="distractor")
result = converter.convert("hotpotqa/hotpot_qa")
```

## Future Enhancements

1. **Caching Layer**: Cache frequently accessed datasets
2. **Remote Datasets**: Support loading from URLs/S3
3. **Incremental Updates**: Support dataset versioning and updates
4. **Advanced Filtering**: SQL-like query interface
5. **Format Support**: Add Parquet, Arrow formats
6. **Distributed Loading**: Support for distributed data loading

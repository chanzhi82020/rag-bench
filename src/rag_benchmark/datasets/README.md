# RAG Benchmark Dataset Module

This module provides standardized dataset management for RAG (Retrieval-Augmented Generation) evaluation, including Golden Dataset loading, conversion, and validation capabilities.

## Features

- **Standardized Data Format**: JSONL-based Golden Dataset format with consistent schema
- **Multiple Dataset Support**: Built-in support for HotpotQA, Natural Questions, and custom datasets
- **Data Validation**: Comprehensive format and quality validation
- **Streaming Support**: Efficient handling of large datasets with streaming loading
- **Conversion Tools**: Convert public datasets to Golden Dataset format
- **Extensible Architecture**: Easy to add new dataset types and converters

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Or install specific packages
pip install pydantic tqdm datasets
```

### Basic Usage

```python
from rag_benchmark.datasets import (
    load_golden_dataset,
    list_golden_datasets,
    validate_dataset
)

# List available datasets
datasets = list_golden_datasets()
print(datasets)

# Load golden records
for record in load_golden_dataset("hotpotqa"):
    print(f"Question: {record.user_input}")
    print(f"Answer: {record.reference}")
    break  # Show first record

# Validate a dataset
result = validate_dataset("path/to/dataset")
print(f"Valid: {result.is_valid}")
```

## Data Format

### Golden Dataset Structure

A Golden Dataset consists of three main files:

1. **qac.jsonl**: Question-Answer-Context records
2. **corpus.jsonl**: Document corpus
3. **metadata.json**: Dataset metadata

#### qac.jsonl Format
```json
{"user_input": "What is the capital of France?", "reference": "Paris", "reference_contexts": ["France is a country...", "Paris has been..."], "reference_context_ids": ["ctx_001", "ctx_002"]}
{"user_input": "Who wrote Romeo and Juliet?", "reference": "William Shakespeare", "reference_contexts": ["Romeo and Juliet is..."], "reference_context_ids": ["ctx_003"]}
```

#### corpus.jsonl Format
```json
{"reference_context": "France is a country in Western Europe...", "reference_context_id": "ctx_001", "title": "France - Wikipedia"}
{"reference_context": "Paris has been the political center...", "reference_context_id": "ctx_002", "title": "Paris - Wikipedia"}
```

#### metadata.json Format
```json
{
  "name": "my_dataset",
  "version": "1.0",
  "description": "Example dataset",
  "source": "internal",
  "size": 100,
  "domain": "general",
  "language": "en"
}
```

## Dataset Management

### Loading Datasets

```python
from rag_benchmark.datasets import load_golden_dataset, load_corpus_dataset

# Stream golden records (memory efficient)
for record in load_golden_dataset("hotpotqa", streaming=True):
    process(record)

# Load all records into memory
records = list(load_golden_dataset("hotpotqa", streaming=False))

# Load corpus documents
for doc in load_corpus_dataset("hotpotqa"):
    print(doc.title)
```

### Listing Datasets

```python
from rag_benchmark.datasets import list_golden_datasets, get_dataset_metadata

# List all registered datasets
all_datasets = list_golden_datasets()

# List only available (locally present) datasets
available = list_golden_datasets(available_only=True)

# Get metadata for a specific dataset
metadata = get_dataset_metadata("hotpotqa")
print(metadata)
```

### Validating Datasets

```python
from rag_benchmark.datasets.validators import validate_dataset

# Validate a dataset
result = validate_dataset("path/to/dataset")

if result.is_valid:
    print("Dataset is valid!")
else:
    print("Dataset has errors:")
    for error in result.errors:
        print(f"  - {error}")

# Check statistics
print("Statistics:", result.statistics)
```

## Converting Datasets

### Convert HotpotQA

```python
from rag_benchmark.datasets import convert_dataset

# Convert HotpotQA dataset
result = convert_dataset(
    source_path="hotpotqa/hotpot_qa",
    output_dir="output/hotpotqa",
    converter_type="hotpotqa",
    variant="distractor"
)

print(f"Converted {result.converted_records} records")
```

### Convert Natural Questions

```python
from rag_benchmark.datasets import create_nq_converter

# Create NQ converter
converter = create_nq_converter(
    output_dir="output/nq",
    subset="validation"
)

# Convert dataset
result = converter.convert("google-research-datasets/natural_questions")
```

### Create Custom Dataset

```python
from rag_benchmark.datasets.loaders import JSONLLoader
from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord

# Create custom loader
loader = JSONLLoader("path/to/custom/dataset")

# Save your data
loader.save_golden_records(your_golden_records)
loader.save_corpus_records(your_corpus_records)
```

## Data Quality

The dataset module provides comprehensive quality validation:

- **Format Validation**: Ensures data follows the correct JSONL format
- **Schema Validation**: Validates required fields and data types
- **Quality Metrics**: Analyzes question/answer quality, context coverage, and relevance
- **Consistency Checks**: Verifies corpus integrity and reference consistency

## Examples

See the `examples/` directory for detailed examples:

- `load_golden_dataset.py`: Loading and using datasets
- `convert_custom_dataset.py`: Converting datasets and creating custom data

## API Reference

### Core Classes

- `GoldenRecord`: A single Q&A record with contexts
- `CorpusRecord`: A single document in the corpus
- `DatasetMetadata`: Metadata about a dataset
- `ValidationResult`: Result of dataset validation

### Main Functions

- `load_golden_dataset(name, streaming=True)`: Load golden records
- `list_golden_datasets(available_only=False)`: List datasets
- `validate_dataset(path)`: Validate a dataset
- `convert_dataset(...)`: Convert datasets to Golden format

### Loaders

- `JSONLLoader`: Load JSONL format datasets
- `BaseLoader`: Abstract base for custom loaders

### Converters

- `HotpotQAConverter`: Convert HotpotQA dataset
- `NaturalQuestionsConverter`: Convert Natural Questions dataset
- `BaseConverter`: Base class for custom converters

## Extending the Module

### Adding New Loaders

```python
from rag_benchmark.datasets.loaders.base import BaseLoader


class MyCustomLoader(BaseLoader):
    def load_golden_records(self):
        # Your implementation
        pass

    def load_corpus_records(self):
        # Your implementation
        pass
```

### Adding New Converters

```python
from scripts.converters import BaseConverter


class MyConverter(BaseConverter):
    def load_source_data(self, source_path):
        # Load your source data
        pass

    def convert_record(self, source_record):
        # Convert to GoldenRecord
        pass
```

## Best Practices

1. **Use Streaming**: For large datasets, always use `streaming=True` to avoid memory issues
2. **Validate Data**: Always validate datasets before using them in evaluation
3. **Check Quality**: Review quality metrics to ensure dataset suitability
4. **Version Control**: Track dataset versions for reproducibility
5. **Document Sources**: Always include source information in metadata

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Memory Issues**: Use streaming mode for large datasets
3. **Validation Errors**: Check file structure and data format
4. **Conversion Failures**: Verify source data format and converter parameters

### Getting Help

- Check the examples directory for usage patterns
- Review the API documentation for detailed function signatures
- Use validation to diagnose data issues
- Enable debug logging for troubleshooting

## License

This module is part of the RAG Benchmark framework. See the project license for details.
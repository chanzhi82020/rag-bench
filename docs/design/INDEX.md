# RAG Benchmark Design Documentation Index

## 📚 Complete Documentation Structure

### Main Overview
- **[README.md](README.md)** - System architecture, data flow, and design principles

---

## 📦 Datasets Module

### Core Documentation
- **[datasets/overview.md](datasets/overview.md)** - Module architecture and design
  - Data schemas (GoldenRecord, CorpusRecord, DatasetMetadata)
  - GoldenDataset interface and operations
  - Dataset registry and management
  - Loaders and validators
  - Design principles and patterns

### Detailed Guides
- **[datasets/converters.md](datasets/converters.md)** - Format analysis and conversion logic
  - HotpotQA format and conversion (multi-hop reasoning)
  - Natural Questions format and conversion (HTML parsing)
  - XQuAD format and conversion (multilingual SQuAD)
  - Conversion challenges and solutions
  - Step-by-step conversion logic with examples

- **[datasets/dataset-comparison.md](datasets/dataset-comparison.md)** - Dataset comparison and quick reference
  - Dataset characteristics comparison
  - Use case recommendations
  - Conversion statistics and performance
  - Quality considerations
  - Troubleshooting guide

---

## 🔧 Prepare Module

### Core Documentation
- **[prepare/overview.md](prepare/overview.md)** - Preparation pipeline design
  - RAGInterface abstract interface
  - Data models (RetrievalResult, GenerationResult, RAGConfig)
  - prepare_experiment_dataset() function
  - Example implementations (DummyRAG, SimpleRAG, BaselineRAG)
  - Batch processing and optimization
  - Integration patterns

---

## 📊 Evaluate Module

### Core Documentation
- **[evaluate/overview.md](evaluate/overview.md)** - Evaluation system design
  - Evaluation functions (e2e, retrieval, generation)
  - RAGAS metrics (LLM-based semantic evaluation)
  - Traditional IR metrics (statistical evaluation)
  - Metric integration and adapters
  - Data requirements and model configuration
  - Cost considerations

---

## 📈 Analysis Module

### Core Documentation
- **[analysis/overview.md](analysis/overview.md)** - Result analysis and visualization
  - ResultComparison class
  - Comparison statistics and DataFrame structure
  - Visualization functions (metrics, comparison, distribution)
  - Statistical analysis methods
  - Error analysis and worst case identification
  - Usage patterns and customization

---

## 🌐 API Module

### Core Documentation
- **[api/overview.md](api/overview.md)** - Web API service design
  - FastAPI application structure
  - State management and persistence
  - Pydantic data models
  - API endpoints (models, datasets, RAG, evaluation)
  - Task execution and progress tracking
  - Model registry pattern
  - Security and performance considerations

---

## 🎯 Quick Navigation by Topic

### Architecture & Design
- [System Architecture](README.md#system-architecture)
- [Data Flow](README.md#data-flow)
- [Design Principles](README.md#design-principles)

### Data Management
- [Data Schemas](datasets/overview.md#core-components)
- [Dataset Loading](datasets/overview.md#goldenDataset-interface)
- [Dataset Conversion](datasets/converters.md)
- [Dataset Comparison](datasets/dataset-comparison.md)

### RAG Integration
- [RAG Interface](prepare/overview.md#raginterface)
- [Baseline RAG Implementation](prepare/overview.md#example-implementations)
- [Custom RAG Integration](prepare/overview.md#integration-patterns)

### Evaluation
- [RAGAS Metrics](evaluate/overview.md#ragas-metrics)
- [Traditional IR Metrics](evaluate/overview.md#traditional-ir-metrics)
- [Metric Selection](evaluate/overview.md#conditional-metric-selection)

### Analysis & Visualization
- [Result Comparison](analysis/overview.md#resultcomparison)
- [Statistical Analysis](analysis/overview.md#statistical-analysis)
- [Visualization](analysis/overview.md#visualization-functions)

### API & Web Service
- [API Endpoints](api/overview.md#api-endpoints)
- [Model Registry](api/overview.md#model-management)
- [Task Management](api/overview.md#task-execution-flow)

---

## 📖 Reading Paths

### For New Users
1. Start with [README.md](README.md) for system overview
2. Read [datasets/overview.md](datasets/overview.md) to understand data structures
3. Review [prepare/overview.md](prepare/overview.md) for RAG integration
4. Check [evaluate/overview.md](evaluate/overview.md) for evaluation basics

### For Dataset Integration
1. [datasets/overview.md](datasets/overview.md) - Understand the data model
2. [datasets/converters.md](datasets/converters.md) - Learn conversion patterns
3. [datasets/dataset-comparison.md](datasets/dataset-comparison.md) - Compare with existing datasets

### For RAG System Integration
1. [prepare/overview.md](prepare/overview.md) - Understand RAGInterface
2. [prepare/overview.md#integration-patterns](prepare/overview.md#integration-patterns) - See integration examples
3. [evaluate/overview.md](evaluate/overview.md) - Plan evaluation strategy

### For API Development
1. [api/overview.md](api/overview.md) - Understand API architecture
2. [api/overview.md#api-endpoints](api/overview.md#api-endpoints) - Review endpoints
3. [api/overview.md#model-management](api/overview.md#model-management) - Learn model registry

### For Result Analysis
1. [evaluate/overview.md](evaluate/overview.md) - Understand evaluation output
2. [analysis/overview.md](analysis/overview.md) - Learn comparison methods
3. [analysis/overview.md#visualization-functions](analysis/overview.md#visualization-functions) - Create visualizations

---

## 🔍 Search by Keyword

### Data Structures
- **GoldenRecord**: [datasets/overview.md](datasets/overview.md#core-components)
- **CorpusRecord**: [datasets/overview.md](datasets/overview.md#core-components)
- **SingleTurnSample**: [README.md](README.md#key-data-structures)
- **EvaluationResult**: [evaluate/overview.md](evaluate/overview.md#results-format)

### Interfaces
- **RAGInterface**: [prepare/overview.md](prepare/overview.md#raginterface)
- **BaseLoader**: [datasets/overview.md](datasets/overview.md#loaders)
- **BaseConverter**: [datasets/overview.md](datasets/overview.md#converters)

### Functions
- **prepare_experiment_dataset**: [prepare/overview.md](prepare/overview.md#preparation-pipeline)
- **evaluate_e2e**: [evaluate/overview.md](evaluate/overview.md#evaluation-functions)
- **compare_results**: [analysis/overview.md](analysis/overview.md#compare_results-function)

### Patterns
- **Batch Processing**: [prepare/overview.md](prepare/overview.md#batch-optimization)
- **Error Handling**: [prepare/overview.md](prepare/overview.md#error-handling)
- **Lazy Loading**: [datasets/overview.md](datasets/overview.md#goldenDataset-interface)

---

## 📝 Document Status

| Document | Status | Last Updated | Completeness |
|----------|--------|--------------|--------------|
| README.md | ✅ Complete | 2024 | 100% |
| datasets/overview.md | ✅ Complete | 2024 | 100% |
| datasets/converters.md | ✅ Complete | 2024 | 100% |
| datasets/dataset-comparison.md | ✅ Complete | 2024 | 100% |
| prepare/overview.md | ✅ Complete | 2024 | 100% |
| evaluate/overview.md | ✅ Complete | 2024 | 100% |
| analysis/overview.md | ✅ Complete | 2024 | 100% |
| api/overview.md | ✅ Complete | 2024 | 100% |

---

## 🤝 Contributing to Documentation

### Adding New Documentation
1. Follow the existing structure and style
2. Include code examples and diagrams
3. Add cross-references to related documents
4. Update this index file

### Documentation Standards
- Use clear, concise language
- Include practical examples
- Explain design rationale
- Add troubleshooting sections
- Keep code examples up-to-date

### Feedback
- Open issues for documentation improvements
- Suggest new topics or sections
- Report outdated information
- Share usage examples

---

## 📚 External References

### Related Technologies
- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### Dataset Sources
- [HotpotQA](https://hotpotqa.github.io/)
- [Natural Questions](https://ai.google.com/research/pubs/pub47761)
- [XQuAD](https://github.com/google-deepmind/xquad)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)

### Research Papers
- [HotpotQA Paper](https://arxiv.org/abs/1809.09600)
- [Natural Questions Paper](https://ai.google.com/research/pubs/pub47761)
- [XQuAD Paper](https://arxiv.org/abs/1910.11856)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)

# Dataset Comparison and Quick Reference

## Supported Datasets Overview

| Dataset | Language | Domain | Size | Question Type | Complexity |
|---------|----------|--------|------|---------------|------------|
| HotpotQA | English | General Knowledge | 113k | Multi-hop reasoning | High |
| Natural Questions | English | Web Search | 307k | Factoid | Medium |
| XQuAD | Multilingual | General Knowledge | 1.2k per language | Extractive | Low |

## Dataset Characteristics

### HotpotQA

**Source**: https://hotpotqa.github.io/

**Characteristics**:
- Multi-hop reasoning questions
- Requires combining information from multiple paragraphs
- Two variants: distractor (10 paragraphs) and fullwiki (full Wikipedia)
- Explicit supporting facts annotations

**Question Types**:
- **Bridge**: "What is the birthplace of the director of Titanic?"
  - Requires: Film → Director → Birthplace
- **Comparison**: "Were Scott Derrickson and Ed Wood of the same nationality?"
  - Requires: Person A → Nationality, Person B → Nationality → Compare

**Difficulty Levels**: easy, medium, hard

**Example**:
```
Question: "Were Scott Derrickson and Ed Wood of the same nationality?"
Answer: "yes"
Supporting Facts:
  - Scott Derrickson, sentence 0: "Scott Derrickson is an American director..."
  - Ed Wood, sentence 0: "Edward D. Wood Jr. was an American filmmaker..."
```

**Conversion Complexity**: ⭐⭐⭐⭐⭐
- Supporting facts may not match context titles exactly
- Bridge questions require entity resolution
- Need to handle both gold and distractor paragraphs

### Natural Questions

**Source**: https://ai.google.com/research/pubs/pub47761

**Characteristics**:
- Real user queries from Google Search
- Answers from Wikipedia articles
- Both short answers (spans) and long answers (paragraphs)
- HTML-formatted documents

**Answer Types**:
- **Short Answer**: Specific span (e.g., "August 19, 2004")
- **Long Answer**: Paragraph containing the answer
- **Yes/No**: Boolean answers for some questions

**Example**:
```
Question: "when did google become a public company"
Short Answer: "August 19, 2004"
Long Answer: "Google became public on August 19, 2004, through an initial public offering..."
```

**Conversion Complexity**: ⭐⭐⭐
- HTML parsing required
- Token span to text conversion
- Need to extract relevant passages from full documents

### XQuAD

**Source**: https://github.com/google-deepmind/xquad

**Characteristics**:
- Cross-lingual Question Answering Dataset
- SQuAD format (extractive QA)
- Multiple languages: ar, de, el, en, es, hi, ru, th, tr, vi, zh
- High-quality human translations

**Languages Supported**:
- Chinese (zh): 1,190 questions
- English (en): 1,190 questions
- Spanish (es): 1,190 questions
- And 8 more languages

**Example (Chinese)**:
```
Question: "法国的首都是哪里？" (What is the capital of France?)
Answer: "巴黎" (Paris)
Context: "巴黎是法国的首都和最大城市..." (Paris is the capital and largest city of France...)
```

**Conversion Complexity**: ⭐⭐
- Standard SQuAD format
- Multiple Q&A pairs per paragraph
- Need global paragraph counter for unique IDs

## Conversion Statistics

### Processing Time (Approximate)

| Dataset | Records | Conversion Time | Output Size |
|---------|---------|-----------------|-------------|
| HotpotQA (distractor) | 90k | ~30 min | ~500 MB |
| Natural Questions (validation) | 8k | ~10 min | ~100 MB |
| XQuAD (zh) | 1.2k | ~1 min | ~5 MB |

*Note: Times are approximate and depend on hardware*

### Output Statistics

**HotpotQA (distractor)**:
- Golden records: ~90,000
- Corpus records: ~900,000 (10 per question)
- Avg question length: 18 words
- Avg answer length: 2 words
- Avg contexts per record: 2

**Natural Questions (validation)**:
- Golden records: ~8,000
- Corpus records: ~8,000-16,000
- Avg question length: 9 words
- Avg answer length: 3 words
- Avg contexts per record: 1-2

**XQuAD (zh)**:
- Golden records: ~1,190
- Corpus records: ~500 (shared across questions)
- Avg question length: 12 characters
- Avg answer length: 4 characters
- Avg contexts per record: 1

## Data Quality Considerations

### HotpotQA

**Strengths**:
- Explicit supporting facts (gold labels)
- Diverse question types
- Challenging multi-hop reasoning

**Limitations**:
- Distractor paragraphs may be too easy to filter
- Some supporting facts may be incomplete
- Bridge questions can be ambiguous

**Quality Metrics**:
- Supporting facts coverage: ~95%
- Answer in context: ~98%
- Multi-hop required: ~100%

### Natural Questions

**Strengths**:
- Real user queries (natural language)
- High-quality Wikipedia content
- Both short and long answers

**Limitations**:
- HTML parsing can be noisy
- Some questions have no answer
- Long answers can be very long

**Quality Metrics**:
- Has short answer: ~50%
- Has long answer: ~80%
- Answer in context: ~100%

### XQuAD

**Strengths**:
- High-quality translations
- Consistent format across languages
- Good for multilingual evaluation

**Limitations**:
- Small dataset size
- Extractive only (no reasoning)
- Limited domain coverage

**Quality Metrics**:
- Answer in context: ~100%
- Translation quality: High
- Cross-lingual consistency: High

## Use Case Recommendations

### When to Use HotpotQA

✅ **Good for**:
- Testing multi-hop reasoning
- Evaluating complex RAG systems
- Benchmarking retrieval quality
- Research on reasoning

❌ **Not ideal for**:
- Simple factoid QA
- Single-hop retrieval
- Quick prototyping (too large)

### When to Use Natural Questions

✅ **Good for**:
- Real-world query patterns
- Web search scenarios
- Diverse question types
- Production-like evaluation

❌ **Not ideal for**:
- Structured reasoning
- Multi-hop questions
- Clean, simple formats

### When to Use XQuAD

✅ **Good for**:
- Multilingual evaluation
- Cross-lingual comparison
- Quick prototyping (small size)
- Extractive QA testing

❌ **Not ideal for**:
- Complex reasoning
- Large-scale benchmarking
- English-only systems

## Conversion Command Reference

### HotpotQA

```python
from scripts.converters import HotpotQAConverter

# Distractor variant
converter = HotpotQAConverter(
    output_dir="data/hotpotqa/distractor",
    variant="distractor"
)
result = converter.convert("hotpotqa/hotpot_qa")

# Fullwiki variant
converter = HotpotQAConverter(
    output_dir="data/hotpotqa/fullwiki",
    variant="fullwiki"
)
result = converter.convert("hotpotqa/hotpot_qa")
```

### Natural Questions

```python
from scripts.converters import NaturalQuestionsConverter

# Validation split
converter = NaturalQuestionsConverter(
    output_dir="data/nq",
    subset="validation"
)
result = converter.convert("google-research-datasets/natural_questions")
```

### XQuAD

```python
from scripts.converters import XQuADConverter

# Chinese
converter = XQuADConverter(
    output_dir="data/xquad/zh",
    language="zh"
)
result = converter.convert("path/to/xquad.zh.json")

# English
converter = XQuADConverter(
    output_dir="data/xquad/en",
    language="en"
)
result = converter.convert("path/to/xquad.en.json")
```

## Troubleshooting Common Issues

### HotpotQA

**Issue**: Supporting facts not found
```
Solution: Check title matching logic
- Verify context titles match supporting fact titles
- Check for entity name variations
- Review entity_to_doc_id mapping
```

**Issue**: Too many corpus records
```
Solution: Enable deduplication
- Use global corpus_map
- Check for duplicate titles
- Verify hash function consistency
```

### Natural Questions

**Issue**: HTML parsing errors
```
Solution: Improve cleaning
- Update regex patterns
- Handle edge cases (nested tags)
- Validate cleaned text
```

**Issue**: Missing answers
```
Solution: Check annotation structure
- Verify short_answers field
- Fallback to long_answer
- Log records without answers
```

### XQuAD

**Issue**: Duplicate corpus records
```
Solution: Use global counter
- Initialize _paragraph_counter = 0
- Increment for each paragraph
- Don't reset between articles
```

**Issue**: Encoding errors
```
Solution: Specify UTF-8
- Use encoding='utf-8' in file operations
- Verify source file encoding
- Check JSON parsing
```

## Performance Optimization Tips

### For Large Datasets (HotpotQA, NQ)

1. **Increase batch size**:
   ```python
   converter = HotpotQAConverter(output_dir="...", batch_size=5000)
   ```

2. **Use streaming**:
   ```python
   # Don't load entire dataset into memory
   for record in converter.load_source_data(source_path):
       process(record)
   ```

3. **Parallel processing** (future enhancement):
   ```python
   # Process multiple records in parallel
   with multiprocessing.Pool(4) as pool:
       results = pool.map(converter.convert_record, records)
   ```

### For Small Datasets (XQuAD)

1. **Reduce batch size** for faster feedback:
   ```python
   converter = XQuADConverter(output_dir="...", batch_size=100)
   ```

2. **Disable progress bar** for cleaner output:
   ```python
   # In convert() method, set disable=True for tqdm
   ```

## Dataset Selection Decision Tree

```
Start
  │
  ├─ Need multilingual? ──Yes──> XQuAD
  │                       
  ├─ Need multi-hop reasoning? ──Yes──> HotpotQA
  │                              
  ├─ Need real user queries? ──Yes──> Natural Questions
  │                           
  ├─ Need large dataset? ──Yes──> HotpotQA or Natural Questions
  │                       
  └─ Quick prototyping? ──Yes──> XQuAD
```

## References

- [HotpotQA Paper](https://arxiv.org/abs/1809.09600)
- [Natural Questions Paper](https://ai.google.com/research/pubs/pub47761)
- [XQuAD Paper](https://arxiv.org/abs/1910.11856)
- [SQuAD Format Specification](https://rajpurkar.github.io/SQuAD-explorer/)

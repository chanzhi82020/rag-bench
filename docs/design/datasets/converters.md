# Dataset Converters - Format Analysis and Conversion Logic

## Overview

This document provides detailed analysis of source dataset formats and the conversion logic to transform them into the Golden Dataset format.

## Golden Dataset Format (Target)

### qac.jsonl (Question-Answer-Context)
```json
{
  "user_input": "What is the capital of France?",
  "reference": "Paris",
  "reference_contexts": ["France is a country...", "Paris has been..."],
  "reference_context_ids": ["ctx_001", "ctx_002"],
  "metadata": {"source": "...", "id": "..."}
}
```

### corpus.jsonl
```json
{
  "reference_context": "France is a country in Western Europe...",
  "reference_context_id": "ctx_001",
  "title": "France - Wikipedia",
  "metadata": {"source": "...", "original_index": 0}
}
```

## HotpotQA Dataset

### Source Format

HotpotQA provides multi-hop reasoning questions with supporting facts.

**Original Structure**:
```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison",
  "level": "medium",
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ],
  "context": [
    ["Scott Derrickson", ["Scott Derrickson is an American director...", "He directed..."]],
    ["Ed Wood", ["Edward D. Wood Jr. was an American filmmaker...", "He is known for..."]]
  ]
}
```


**Field Descriptions**:
- `_id`: Unique question identifier
- `question`: The multi-hop question
- `answer`: Ground truth answer (string)
- `type`: Question type (bridge, comparison)
- `level`: Difficulty (easy, medium, hard)
- `supporting_facts`: List of [title, sentence_id] pairs indicating gold evidence
- `context`: List of [title, sentences_list] providing candidate paragraphs

**Variants**:
- **distractor**: 10 paragraphs (2 gold + 8 distractors)
- **fullwiki**: Full Wikipedia as corpus (retrieval required)

### Conversion Challenges

**Challenge 1: Supporting Facts Resolution**

Supporting facts reference paragraphs by title, but titles may not exactly match context titles.

**Example**:
```python
supporting_facts = [["Scott Derrickson", 0], ["Ed Wood", 0]]
context = [
    ["Scott Derrickson", [...]],
    ["Ed Wood (film)", [...]]  # Title mismatch!
]
```

**Solution**: Multi-level matching strategy
1. Exact title match
2. Entity name mapping (heuristic)
3. Content-based matching (paragraph mentions entity)

**Challenge 2: Bridge Questions**

Bridge questions require information from multiple paragraphs that may reference each other indirectly.

**Example**:
```
Question: "What is the birthplace of the director of Titanic?"
Supporting facts:
- ["Titanic (1997 film)", 0] → "directed by James Cameron"
- ["James Cameron", 2] → "born in Kapuskasing, Ontario"
```

The second paragraph may not have "James Cameron" as title but contains the information.

**Solution**: Entity-based mapping
```python
# Build entity → doc_id mapping
for title, sentences in context:
    full_text = " ".join(sentences)
    for sf_title, _ in supporting_facts:
        if sf_title.lower() in full_text.lower():
            entity_to_doc_id[sf_title] = doc_id
```

### Conversion Logic

```python
def convert_record(self, source_record: Dict[str, Any]) -> List[Tuple[GoldenRecord, List[CorpusRecord]]]:
    question = source_record["question"]
    answer = source_record["answer"]
    supporting_facts = source_record["supporting_facts"]  # [[title, sent_id], ...]
    context = source_record["context"]  # [[title, [sentences]], ...]
    
    # Step 1: Build corpus records and title mappings
    corpus_records = []
    title_to_doc_id = {}
    entity_to_doc_id = {}
    
    for idx, (title, sentences) in enumerate(context):
        full_text = " ".join(sentences)
        doc_id = f"hotpot_{variant}_{_id}_{idx}_{hash(title)}"
        
        # Exact title mapping
        title_to_doc_id[title] = doc_id
        
        # Entity mapping (for bridge questions)
        for sf_title, _ in supporting_facts:
            if sf_title.lower() in full_text.lower():
                entity_to_doc_id[sf_title] = doc_id
        
        corpus_records.append(CorpusRecord(
            reference_context=full_text,
            reference_context_id=doc_id,
            title=title,
            metadata={"source": "hotpotqa", "original_index": idx}
        ))
    
    # Step 2: Resolve supporting facts to reference contexts
    reference_contexts = []
    reference_context_ids = []
    
    for sf_title, sent_id in supporting_facts:
        # Try exact match first
        doc_id = title_to_doc_id.get(sf_title)
        
        # Fallback to entity mapping
        if not doc_id:
            doc_id = entity_to_doc_id.get(sf_title)
        
        if doc_id:
            # Find the corpus record
            for corp in corpus_records:
                if corp.reference_context_id == doc_id:
                    reference_contexts.append(corp.reference_context)
                    reference_context_ids.append(doc_id)
                    break
        else:
            # Missing gold document (rare)
            logger.warning(f"Supporting fact not resolved: {sf_title}")
    
    # Step 3: Create GoldenRecord
    golden_record = GoldenRecord(
        user_input=question,
        reference=answer,
        reference_contexts=reference_contexts,
        reference_context_ids=reference_context_ids,
        metadata={
            "source": "hotpotqa",
            "id": _id,
            "type": source_record["type"],
            "level": source_record["level"]
        }
    )
    
    return [(golden_record, corpus_records)]
```

**Key Design Decisions**:
1. **All paragraphs in corpus**: Include both gold and distractor paragraphs
2. **Stable doc_id**: Use hash of title for reproducibility
3. **Multi-level resolution**: Handle title mismatches gracefully
4. **Metadata preservation**: Keep original type, level, id

### Example Conversion

**Input (HotpotQA)**:
```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison",
  "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
  "context": [
    ["Scott Derrickson", ["Scott Derrickson is an American director."]],
    ["Ed Wood", ["Edward D. Wood Jr. was an American filmmaker."]]
  ]
}
```

**Output (Golden Format)**:

qac.jsonl:
```json
{
  "user_input": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "reference": "yes",
  "reference_contexts": [
    "Scott Derrickson is an American director.",
    "Edward D. Wood Jr. was an American filmmaker."
  ],
  "reference_context_ids": ["hotpot_distractor_5a8b57f25542995d1e6f1371_0_a1b2c3d4", 
                            "hotpot_distractor_5a8b57f25542995d1e6f1371_1_e5f6g7h8"],
  "metadata": {"source": "hotpotqa", "id": "5a8b57f25542995d1e6f1371", "type": "comparison"}
}
```

corpus.jsonl:
```json
{"reference_context": "Scott Derrickson is an American director.", 
 "reference_context_id": "hotpot_distractor_5a8b57f25542995d1e6f1371_0_a1b2c3d4",
 "title": "Scott Derrickson", "metadata": {"source": "hotpotqa", "original_index": 0}}
{"reference_context": "Edward D. Wood Jr. was an American filmmaker.",
 "reference_context_id": "hotpot_distractor_5a8b57f25542995d1e6f1371_1_e5f6g7h8",
 "title": "Ed Wood", "metadata": {"source": "hotpotqa", "original_index": 1}}
```

## Natural Questions Dataset

### Source Format

Natural Questions contains real Google search queries with answers from Wikipedia.

**Original Structure**:
```json
{
  "id": "797803103760793766",
  "document": {
    "title": "Google",
    "url": "https://en.wikipedia.org/wiki/Google",
    "html": "<html>...</html>",
    "tokens": {
      "token": ["Google", "is", "an", "American", ...],
      "is_html": [false, false, false, false, ...]
    }
  },
  "question": {
    "text": "when did google become a public company"
  },
  "annotations": [{
    "short_answers": [{
      "start_token": 123,
      "end_token": 125,
      "text": "August 19, 2004"
    }],
    "long_answer": {
      "start_token": 120,
      "end_token": 200,
      "candidate_index": 5
    },
    "yes_no_answer": "NONE"
  }]
}
```

**Field Descriptions**:
- `document`: Wikipedia page with HTML and tokens
- `question.text`: User's natural question
- `annotations.short_answers`: Span-based short answers
- `annotations.long_answer`: Paragraph-level long answer
- `tokens`: Tokenized document with HTML markers

### Conversion Challenges

**Challenge 1: HTML Parsing**

Documents contain HTML markup that needs cleaning.

**Solution**: HTML tag removal
```python
def _clean_text(self, text: str) -> str:
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

**Challenge 2: Token Span to Text**

Answers are specified as token spans, need to extract text.

**Solution**: Token reconstruction
```python
def _extract_answer(self, annotations: Dict) -> str:
    short_answers = annotations.get("short_answers", [])
    if short_answers:
        # Use first short answer
        return short_answers[0].get("text", "")
    
    # Fallback to long answer
    long_answer = annotations.get("long_answer", {})
    return long_answer.get("text", "")
```

**Challenge 3: Passage Extraction**

Need to extract relevant passage from full document.

**Solution**: Use long_answer span
```python
def _extract_relevant_passages(self, document: Dict, annotations: Dict) -> List[Tuple[str, str]]:
    long_answer = annotations.get("long_answer", {})
    start_token = long_answer.get("start_token", -1)
    end_token = long_answer.get("end_token", -1)
    
    if start_token >= 0 and end_token > start_token:
        tokens = document.get("tokens", {}).get("token", [])
        passage_tokens = tokens[start_token:end_token]
        passage_text = " ".join(passage_tokens)
        
        passage_id = f"passage_{hash(doc_url + str(start_token))}"
        return [(passage_id, passage_text)]
    
    # Fallback: use entire document (truncated)
    html = document.get("html", "")
    passage_text = self._clean_text(html[:1000])
    passage_id = f"doc_{hash(doc_url)}"
    return [(passage_id, passage_text)]
```

### Conversion Logic

```python
def convert_record(self, source_record: Dict[str, Any]) -> List[Tuple[GoldenRecord, List[CorpusRecord]]]:
    # Extract question
    question = source_record.get("question", {}).get("text", "")
    question = self._clean_text(question)
    
    # Extract answer from annotations
    annotations = source_record.get("annotations", [])
    if not annotations:
        raise ValueError("No annotations")
    
    answer = self._extract_answer(annotations[0])
    if not answer:
        raise ValueError("No answer found")
    
    # Extract document
    document = source_record.get("document", {})
    title = document.get("title", "Unknown")
    doc_url = document.get("url", "")
    
    # Extract relevant passages
    passages = self._extract_relevant_passages(document, annotations[0])
    
    # Build corpus records
    corpus_records = []
    reference_contexts = []
    reference_context_ids = []
    
    for passage_id, passage_text in passages:
        reference_contexts.append(passage_text)
        reference_context_ids.append(passage_id)
        
        corpus_records.append(CorpusRecord(
            reference_context=passage_text,
            reference_context_id=passage_id,
            title=title,
            metadata={
                "source": "natural_questions",
                "document_url": doc_url
            }
        ))
    
    # Create GoldenRecord
    golden_record = GoldenRecord(
        user_input=question,
        reference=answer,
        reference_contexts=reference_contexts,
        reference_context_ids=reference_context_ids,
        metadata={
            "source": "natural_questions",
            "id": source_record.get("id", ""),
            "document_url": doc_url
        }
    )
    
    return [(golden_record, corpus_records)]
```

**Key Design Decisions**:
1. **Prefer short answers**: More concise and specific
2. **Clean HTML**: Remove markup for readability
3. **Extract passages**: Use long_answer span when available
4. **Fallback strategy**: Use full document if spans unavailable

### Example Conversion

**Input (Natural Questions)**:
```json
{
  "id": "797803103760793766",
  "document": {
    "title": "Google",
    "url": "https://en.wikipedia.org/wiki/Google",
    "tokens": {"token": ["Google", "became", "public", "on", "August", "19", ",", "2004", "."]}
  },
  "question": {"text": "when did google become a public company"},
  "annotations": [{
    "short_answers": [{"text": "August 19, 2004"}],
    "long_answer": {"start_token": 0, "end_token": 9}
  }]
}
```

**Output (Golden Format)**:

qac.jsonl:
```json
{
  "user_input": "when did google become a public company",
  "reference": "August 19, 2004",
  "reference_contexts": ["Google became public on August 19, 2004."],
  "reference_context_ids": ["passage_a1b2c3d4"],
  "metadata": {"source": "natural_questions", "id": "797803103760793766"}
}
```

corpus.jsonl:
```json
{
  "reference_context": "Google became public on August 19, 2004.",
  "reference_context_id": "passage_a1b2c3d4",
  "title": "Google",
  "metadata": {"source": "natural_questions", "document_url": "https://en.wikipedia.org/wiki/Google"}
}
```

## XQuAD Dataset

### Source Format

XQuAD is a multilingual QA dataset in SQuAD format.

**Original Structure**:
```json
{
  "data": [
    {
      "title": "巴黎",
      "paragraphs": [
        {
          "context": "巴黎是法国的首都和最大城市...",
          "qas": [
            {
              "id": "56ddde6b9a695914005b9628",
              "question": "法国的首都是哪里？",
              "answers": [
                {"text": "巴黎", "answer_start": 0}
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**Field Descriptions**:
- `data`: List of articles
- `title`: Article title
- `paragraphs`: List of paragraphs in the article
- `context`: Paragraph text
- `qas`: Questions about this paragraph
- `answers`: List of answer spans (may have multiple)

### Conversion Challenges

**Challenge 1: Multiple Q&A per Paragraph**

One paragraph may have multiple questions.

**Solution**: Create separate GoldenRecord for each Q&A pair
```python
for qa in paragraph["qas"]:
    # Create one GoldenRecord per question
    golden_record = GoldenRecord(...)
```

**Challenge 2: Corpus Deduplication**

Same paragraph referenced by multiple questions.

**Solution**: Use global paragraph counter for unique IDs
```python
self._paragraph_counter = 0

def convert_record(self, source_record):
    for paragraph in paragraphs:
        paragraph_id = f"xquad_{language}_{self._paragraph_counter:04d}"
        self._paragraph_counter += 1
```

**Challenge 3: Multiple Answers**

XQuAD may have multiple valid answers per question.

**Solution**: Use first answer as reference
```python
answers = qa.get("answers", [])
if answers:
    answer = answers[0].get("text", "")
```

### Conversion Logic

```python
def convert_record(self, source_record: Dict[str, Any]) -> List[Tuple[GoldenRecord, List[CorpusRecord]]]:
    results = []
    
    title = source_record.get("title", "")
    paragraphs = source_record.get("paragraphs", [])
    
    for paragraph in paragraphs:
        context = paragraph.get("context", "")
        qas = paragraph.get("qas", [])
        
        # Create unique corpus record for this paragraph
        paragraph_id = f"xquad_{self.language}_{self._paragraph_counter:04d}"
        self._paragraph_counter += 1
        
        corpus_record = CorpusRecord(
            reference_context=context,
            reference_context_id=paragraph_id,
            title=title,
            metadata={
                "source": "xquad",
                "language": self.language,
                "article_title": title
            }
        )
        
        # Create GoldenRecord for each Q&A pair
        for qa in qas:
            question = qa.get("question", "")
            answers = qa.get("answers", [])
            qa_id = qa.get("id", "")
            
            if not question or not answers:
                continue
            
            answer = answers[0].get("text", "")
            answer_start = answers[0].get("answer_start", -1)
            
            golden_record = GoldenRecord(
                user_input=question,
                reference=answer,
                reference_contexts=[context],
                reference_context_ids=[paragraph_id],
                metadata={
                    "source": "xquad",
                    "language": self.language,
                    "qa_id": qa_id,
                    "answer_start": answer_start,
                    "article_title": title
                }
            )
            
            results.append((golden_record, [corpus_record]))
    
    return results
```

**Key Design Decisions**:
1. **One record per Q&A**: Each question gets separate GoldenRecord
2. **Global counter**: Ensures unique paragraph IDs across articles
3. **First answer**: Use first answer when multiple exist
4. **Language tracking**: Store language in metadata

### Example Conversion

**Input (XQuAD Chinese)**:
```json
{
  "data": [{
    "title": "巴黎",
    "paragraphs": [{
      "context": "巴黎是法国的首都和最大城市，位于法国北部。",
      "qas": [
        {
          "id": "5726534a708984140094c2a1",
          "question": "法国的首都是哪里？",
          "answers": [{"text": "巴黎", "answer_start": 0}]
        },
        {
          "id": "5726534a708984140094c2a2",
          "question": "巴黎位于法国的哪个方向？",
          "answers": [{"text": "北部", "answer_start": 18}]
        }
      ]
    }]
  }]
}
```

**Output (Golden Format)**:

qac.jsonl (2 records):
```json
{"user_input": "法国的首都是哪里？", "reference": "巴黎", 
 "reference_contexts": ["巴黎是法国的首都和最大城市，位于法国北部。"],
 "reference_context_ids": ["xquad_zh_0000"],
 "metadata": {"source": "xquad", "language": "zh", "qa_id": "5726534a708984140094c2a1"}}
{"user_input": "巴黎位于法国的哪个方向？", "reference": "北部",
 "reference_contexts": ["巴黎是法国的首都和最大城市，位于法国北部。"],
 "reference_context_ids": ["xquad_zh_0000"],
 "metadata": {"source": "xquad", "language": "zh", "qa_id": "5726534a708984140094c2a2"}}
```

corpus.jsonl (1 record, shared by both questions):
```json
{"reference_context": "巴黎是法国的首都和最大城市，位于法国北部。",
 "reference_context_id": "xquad_zh_0000",
 "title": "巴黎",
 "metadata": {"source": "xquad", "language": "zh", "article_title": "巴黎"}}
```

## Comparison of Converters

| Aspect | HotpotQA | Natural Questions | XQuAD |
|--------|----------|-------------------|-------|
| **Complexity** | High (multi-hop) | Medium (HTML parsing) | Low (SQuAD format) |
| **Main Challenge** | Supporting facts resolution | Token span extraction | Multiple Q&A per paragraph |
| **Corpus Size** | 10 paragraphs per question | 1-2 passages per question | 1 paragraph per question |
| **Answer Type** | String | Short/long answer | Span-based |
| **Languages** | English | English | Multilingual |
| **Special Handling** | Entity mapping for bridge questions | HTML cleaning | Global paragraph counter |

## Common Patterns

### Pattern 1: Stable ID Generation
```python
# Use hash for reproducibility
doc_id = f"{dataset}_{variant}_{record_id}_{idx}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
```

### Pattern 2: Corpus Deduplication
```python
# Global corpus map to avoid duplicates
self.corpus_map: Dict[str, CorpusRecord] = {}

if doc_id not in self.corpus_map:
    self.corpus_map[doc_id] = corpus_record
```

### Pattern 3: Batch Writing
```python
# Write in batches for efficiency
if len(golden_batch) >= self.batch_size:
    self._write_batch(golden_batch, corpus_batch)
    golden_batch = []
    corpus_batch = []
```

### Pattern 4: Error Recovery
```python
try:
    converted = self.convert_record(source_record)
    self.converted_count += 1
except Exception as e:
    self.failed_count += 1
    logger.error(f"Failed to convert record: {e}")
    if not skip_on_error:
        raise
```

## Best Practices

1. **Preserve Original IDs**: Store original dataset IDs in metadata
2. **Stable Hashing**: Use deterministic hashing for reproducibility
3. **Metadata Rich**: Include source, type, and other useful information
4. **Validation**: Validate converted data before writing
5. **Progress Tracking**: Use tqdm for long conversions
6. **Error Logging**: Log failed records for debugging
7. **Batch Processing**: Write in batches for efficiency
8. **Deduplication**: Avoid duplicate corpus records

## Adding New Converters

To add a new dataset converter:

1. **Analyze Source Format**: Understand the structure
2. **Identify Challenges**: Note special cases
3. **Implement BaseConverter**:
   ```python
   class MyConverter(BaseConverter):
       def load_source_data(self, source_path):
           # Load source data
       
       def convert_record(self, source_record):
           # Convert to GoldenRecord + CorpusRecords
       
       def create_metadata(self, source_path, num_records):
           # Create dataset metadata
   ```
4. **Handle Edge Cases**: Missing fields, malformed data
5. **Test Thoroughly**: Verify conversion correctness
6. **Document**: Add to this document

## Troubleshooting

### Issue: Missing Supporting Facts
**Symptom**: reference_contexts is empty
**Solution**: Check title matching logic, add fallback strategies

### Issue: Duplicate Corpus Records
**Symptom**: corpus.jsonl has duplicates
**Solution**: Use global corpus_map for deduplication

### Issue: Encoding Errors
**Symptom**: UnicodeDecodeError
**Solution**: Use `encoding='utf-8'` when reading/writing

### Issue: Memory Issues
**Symptom**: Out of memory during conversion
**Solution**: Reduce batch_size, use streaming

## References

- [HotpotQA Paper](https://arxiv.org/abs/1809.09600)
- [Natural Questions Paper](https://ai.google.com/research/pubs/pub47761)
- [XQuAD Paper](https://arxiv.org/abs/1910.11856)
- [SQuAD Format](https://rajpurkar.github.io/SQuAD-explorer/)

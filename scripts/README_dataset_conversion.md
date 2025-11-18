# Dataset Conversion Examples

## XQuAD Conversion
```bash
# Convert XQuAD from local file
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input data/xquad/xquad.zh.json \
  --output datasets/xquad/zh \
  --dataset-type xquad \
  --language zh \
  --validate

# Convert XQuAD from HuggingFace
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input xquad \
  --output datasets/xquad/en \
  --dataset-type xquad \
  --language en
```

## HotpotQA Conversion
```bash
# Convert HotpotQA (distractor variant) from HuggingFace
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input hotpotqa \
  --output datasets/hotpotqa_distractor \
  --dataset-type hotpotqa \
  --variant distractor \
  --validate

# Convert HotpotQA (fullwiki variant)
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input hotpotqa \
  --output datasets/hotpotqa_fullwiki \
  --dataset-type hotpotqa \
  --variant fullwiki

# Convert from local HotpotQA file
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input data/hotpotqa/hotpot_dev_distractor_v1.json \
  --output datasets/hotpotqa/distractor \
  --dataset-type hotpotqa \
  --variant distractor
  
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input data/hotpotqa/hotpot_dev_fullwiki_v1.json \
  --output datasets/hotpotqa/fullwiki \
  --dataset-type hotpotqa \
  --variant fullwiki
  
```

## Natural Questions Conversion
```bash
# Convert NQ (validation subset) from HuggingFace
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input natural_questions \
  --output datasets/nq_validation \
  --dataset-type nq \
  --subset validation \
  --validate

# Convert NQ (train subset)
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input natural_questions \
  --output datasets/nq_train \
  --dataset-type nq \
  --subset train

# Convert from local NQ file
PYTHONPATH=/home/chenzhi/rag-bench/src uv run python convert_dataset.py \
  --input data/nq/nq_sample.jsonl \
  --output datasets/nq_sample \
  --dataset-type nq \
  --subset validation
```

## Notes
- The script automatically detects whether the input is a local file or HuggingFace dataset
- For XQuAD, the language defaults to Chinese ("zh")
- For HotpotQA, the variant defaults to "distractor"
- For Natural Questions, the subset defaults to "validation"
- Use `--validate` to automatically validate the converted dataset after conversion
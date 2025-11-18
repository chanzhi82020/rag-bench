"""HotpotQA dataset converter to Golden Dataset format"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple, Union

from datasets import load_dataset as hf_load_dataset

from rag_benchmark.datasets.converters.base import BaseConverter
from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord, DatasetMetadata

logger = logging.getLogger(__name__)


class HotpotQAConverter(BaseConverter):
    """Converter for HotpotQA dataset to Golden Dataset format"""

    def __init__(self,
                 output_dir: Union[str, Path],
                 variant: str = "distractor",
                 batch_size: int = 1000):
        """Initialize HotpotQA converter
        
        Args:
            output_dir: Output directory for converted dataset
            variant: HotpotQA variant ("distractor" or "fullwiki")
            batch_size: Number of records to process in each batch
        """
        super().__init__(output_dir, batch_size)
        self.variant = variant

    def load_source_data(self, source_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Load HotpotQA dataset
        
        Args:
            source_path: Can be:
                - HuggingFace dataset identifier (e.g., "hotpot_qa" or "hotpotqa/hotpot_qa")
                - Local file path (JSON/JSONL format)
                
        Returns:
            Iterator of HotpotQA records
        """
        source_path = str(source_path).strip()

        # Check if it's a local file path
        if Path(source_path).exists():
            logger.info(f"Loading HotpotQA from local file: {source_path}")
            return self._load_from_local_file(source_path)

        # Check if it looks like a HuggingFace dataset identifier
        # If it doesn't contain '/', assume it's just the dataset name
        if '/' not in source_path or source_path.startswith('hotpot'):
            logger.info(f"Loading HotpotQA from HuggingFace: {source_path}")
            return self._load_from_huggingface(source_path)

        # If it's a URL-like pattern, try HuggingFace
        if source_path.count('/') == 1:
            logger.info(f"Loading HotpotQA from HuggingFace: {source_path}")
            return self._load_from_huggingface(source_path)

        # Default: try as local file first, then HuggingFace
        logger.info(f"Trying to load {source_path} as local file first...")
        return self._load_from_local_file(source_path)

    def _load_from_huggingface(self, dataset_name: str) -> Iterator[Dict[str, Any]]:
        """Load dataset from HuggingFace"""
        try:
            # Normalize dataset name
            if dataset_name == "hotpot_qa":
                dataset_name = "hotpotqa/hotpot_qa"

            dataset = hf_load_dataset(dataset_name, self.variant)

            # Use train or validation split
            if "validation" in dataset:
                split = "validation"
            elif "train" in dataset:
                split = "train"
            else:
                split = list(dataset.keys())[0]

            logger.info(f"Using split: {split}")

            for record in dataset[split]:
                yield record

        except Exception as e:
            logger.error(f"Failed to load HotpotQA from HuggingFace: {e}")
            raise

    def _load_from_local_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """Load dataset from local file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")

        # Determine file format and load accordingly
        if file_path.suffix.lower() == '.jsonl':
            # JSONL format
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
        elif file_path.suffix.lower() == '.json':
            # JSON format
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    for record in data:
                        yield record
                elif isinstance(data, dict) and 'data' in data:
                    # Assume data is in {'data': [...]} format
                    for record in data['data']:
                        yield record
                elif isinstance(data, dict):
                    # Single record
                    yield data
                else:
                    raise ValueError(f"Unexpected JSON structure in {file_path}")
        else:
            # Try to parse as JSONL by default
            logger.warning(f"Unknown file extension {file_path.suffix}, trying JSONL format...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    def convert_record(self, source_record: Dict[str, Any]) -> List[Tuple[GoldenRecord, List[CorpusRecord]]]:
        """Convert a single HotpotQA record (distractor or fullwiki) to Golden format"""

        question = source_record["question"].strip()
        answer = str(source_record["answer"]).strip()
        supporting_facts = source_record.get("supporting_facts", [])  # List[List[str, int]]
        context = source_record.get("context", [])  # List[List[title, List[sentences]]]
        _id = source_record.get("_id", "")
        qtype = source_record.get("type", "")
        level = source_record.get("level", "")

        try:
            if not question or not answer:
                raise ValueError("question or answer is empty")

            # Step 1: Build all corpus records + title → doc_id mapping
            corpus_records: List[CorpusRecord] = []
            title_to_doc_id: Dict[str, str] = {}  # exact title → doc_id
            entity_to_doc_id: Dict[str, str] = {}  # fallback: entity name → doc_id (for bridge questions)

            for idx, (title, sentences) in enumerate(context):
                if not isinstance(sentences, list):
                    sentences = [str(sentences)]
                full_text = " ".join(sentences).strip()
                if not full_text:
                    continue

                # Generate stable doc_id
                doc_id = f"hotpot_{self.variant}_{_id}_{idx}_{hashlib.md5(title.encode()).hexdigest()[:8]}"

                title_to_doc_id[title] = doc_id

                # Heuristic: if this paragraph mentions any supporting fact entity, map it
                para_lower = full_text.lower()
                title_lower = title.lower()
                for sf_title, _ in supporting_facts:
                    sf_lower = sf_title.lower()
                    if (sf_lower in para_lower or
                            sf_lower.replace(" ", "") in title_lower.replace(" ", "") or
                            sf_lower in title_lower):
                        entity_to_doc_id[sf_title] = doc_id
                        # One entity may appear in multiple docs, we keep the first one (most common practice)

                corpus_records.append(CorpusRecord(
                    reference_context=full_text,
                    reference_context_id=doc_id,
                    title=title,
                    metadata={
                        "source": "hotpotqa",
                        "variant": self.variant,
                        "original_index": idx,
                        "sentence_count": len(sentences)
                    }
                ))

            # Step 2: Determine gold reference contexts (supporting paragraphs)
            reference_contexts: List[str] = []
            reference_context_ids: List[str] = []

            seen_titles = set()
            for sf_title, sent_id in supporting_facts:
                if sf_title in seen_titles:
                    continue
                seen_titles.add(sf_title)

                # Try exact title match first
                doc_id = title_to_doc_id.get(sf_title)

                # If not found → use heuristic entity mapping (handles bridge questions)
                if not doc_id:
                    doc_id = entity_to_doc_id.get(sf_title)

                if doc_id:
                    # Find the full text of this document
                    for corp in corpus_records:
                        if corp.reference_context_id == doc_id:
                            reference_contexts.append(corp.reference_context)
                            reference_context_ids.append(doc_id)
                            break
                else:
                    # Final fallback: still can't find (extremely rare)
                    warning_text = f"[Gold document not found in provided context] Title: {sf_title} (sent_id: {sent_id})"
                    reference_contexts.append(warning_text)
                    fake_id = f"missing_{hashlib.md5(sf_title.encode()).hexdigest()[:8]}"
                    reference_context_ids.append(fake_id)
                    logger.debug(f"Supporting fact title not resolved: {sf_title}")

            # If no supporting facts at all (shouldn't happen in HotpotQA), use empty list
            if not reference_contexts:
                raise ValueError("reference_contexts cannot be empty")

            # Step 3: Build metadata
            metadata = {
                "source": "hotpotqa",
                "variant": self.variant,
                "id": _id,
                "type": qtype,
                "level": level,
                "original_supporting_facts": supporting_facts,
                "context_paragraphs_count": len(context),
                "gold_documents_resolved": len(reference_contexts),
                "gold_documents_missing": len(supporting_facts) - len(reference_contexts)
            }

            # Step 4: Create GoldenRecord
            golden_record = GoldenRecord(
                user_input=question,
                reference=answer,
                reference_contexts=reference_contexts,
                reference_context_ids=reference_context_ids if reference_context_ids != [None] else None,
                metadata=metadata
            )

            return [(golden_record, corpus_records)]

        except Exception as e:
            logger.error(f"Failed to convert HotpotQA record {_id}: {e}")
            raise ValueError(f"Conversion failed: {e}")

    def create_metadata(self, source_path: Union[str, Path], num_records: int) -> DatasetMetadata:
        """Create metadata for the converted dataset
        
        Args:
            source_path: Path to source data
            num_records: Number of converted records
            
        Returns:
            DatasetMetadata object
        """
        return DatasetMetadata(
            name=f"hotpotqa_{self.variant}",
            version="1.0",
            description=f"HotpotQA {self.variant} dataset converted to Golden Dataset format",
            source="https://hotpotqa.github.io/",
            size=num_records,
            domain="general_knowledge",
            language="en",
            created_at=None,
            license="MIT",
            tags=["multi-hop", "question_answering", "reasoning", self.variant]
        )

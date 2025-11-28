"""Natural Questions dataset converter to Golden Dataset format"""

from typing import Iterator, List, Dict, Any, Tuple, Union
from pathlib import Path
import re
import hashlib
import json
import logging
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)

from scripts.converters.base import BaseConverter
from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord, DatasetMetadata


class NaturalQuestionsConverter(BaseConverter):
    """Converter for Natural Questions dataset to Golden Dataset format"""
    
    def __init__(self, 
                 output_dir: Union[str, Path],
                 subset: str = "validation",
                 batch_size: int = 1000):
        """Initialize Natural Questions converter
        
        Args:
            output_dir: Output directory for converted dataset
            subset: Dataset subset to use ("validation" or "train")
            batch_size: Number of records to process in each batch
        """
        super().__init__(output_dir, batch_size)
        self.subset = subset
    
    def load_source_data(self, source_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Load Natural Questions dataset
        
        Args:
            source_path: Can be:
                - HuggingFace dataset identifier (e.g., "natural_questions" or "google-research-datasets/natural_questions")
                - Local file path (JSON/JSONL format)
                
        Returns:
            Iterator of NQ records
        """
        source_path = str(source_path).strip()
        
        # Check if it's a local file path
        if Path(source_path).exists():
            logger.info(f"Loading Natural Questions from local file: {source_path}")
            return self._load_from_local_file(source_path)
        
        # Check if it looks like a HuggingFace dataset identifier
        # Common identifiers for Natural Questions
        if any(name in source_path.lower() for name in ['natural_questions', 'nq', 'google-research-datasets']):
            logger.info(f"Loading Natural Questions from HuggingFace: {source_path}")
            return self._load_from_huggingface(source_path)
        
        # If it's a URL-like pattern, try HuggingFace
        if source_path.count('/') == 1:
            logger.info(f"Loading Natural Questions from HuggingFace: {source_path}")
            return self._load_from_huggingface(source_path)
        
        # Default: try as local file first
        logger.info(f"Trying to load {source_path} as local file first...")
        return self._load_from_local_file(source_path)
    
    def _load_from_huggingface(self, dataset_name: str) -> Iterator[Dict[str, Any]]:
        """Load dataset from HuggingFace"""
        try:
            # Normalize dataset name
            if dataset_name.lower() in ['natural_questions', 'nq']:
                dataset_name = "google-research-datasets/natural_questions"
            
            dataset = hf_load_dataset(dataset_name, "default")
            
            # Use specified subset
            if self.subset in dataset:
                split = self.subset
            else:
                # Try common splits
                for common_split in ["validation", "train", "test"]:
                    if common_split in dataset:
                        split = common_split
                        break
                else:
                    split = list(dataset.keys())[0]
            
            logger.info(f"Using split: {split}")
            
            for record in dataset[split]:
                yield record
                
        except Exception as e:
            logger.error(f"Failed to load Natural Questions from HuggingFace: {e}")
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
                elif isinstance(data, dict):
                    # Check for common NQ structures
                    if 'data' in data:
                        for record in data['data']:
                            yield record
                    elif 'examples' in data:
                        for record in data['examples']:
                            yield record
                    else:
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
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags and extra whitespace
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_answer(self, annotations: Dict[str, Any]) -> str:
        """Extract answer from annotations
        
        Args:
            annotations: NQ annotations
            
        Returns:
            Extracted answer string
        """
        # Try short answers first
        short_answers = annotations.get("short_answers", [])
        if short_answers:
            # Use the first short answer
            answer_text = short_answers[0].get("text", "")
            return self._clean_text(answer_text)
        
        # Try long answer
        long_answer = annotations.get("long_answer", {})
        if long_answer:
            # For long answers, we need the text from the document
            # This is simplified - in practice you'd extract from the document
            answer_text = long_answer.get("text", "")
            if answer_text:
                return self._clean_text(answer_text)
        
        return ""
    
    def _extract_relevant_passages(self, 
                                 document: Dict[str, Any], 
                                 annotations: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract relevant passages from document
        
        Args:
            document: NQ document
            annotations: NQ annotations
            
        Returns:
            List of (passage_id, passage_text) tuples
        """
        passages = []
        
        # Get document HTML
        html = document.get("html", "")
        if not html:
            return passages
        
        # Extract long answer start and end tokens
        long_answer = annotations.get("long_answer", {})
        start_token = long_answer.get("start_token", -1)
        end_token = long_answer.get("end_token", -1)
        
        # If we have a long answer span
        if start_token >= 0 and end_token > start_token:
            # Get tokens
            tokens = document.get("tokens", {})
            token_strings = tokens.get("token", [])
            
            if token_strings and start_token < len(token_strings) and end_token <= len(token_strings):
                # Extract the passage
                passage_tokens = token_strings[start_token:end_token]
                passage_text = " ".join(passage_tokens)
                
                # Generate ID
                doc_id = document.get("document_url", "")
                passage_id = f"passage_{hashlib.md5(f'{doc_id}_{start_token}'.encode()).hexdigest()[:8]}"
                
                passages.append((passage_id, passage_text))
        
        # If no long answer, use the whole document (simplified)
        if not passages:
            # Create a passage from the entire document
            title = document.get("title", "")
            passage_text = f"Title: {title}\n\n{self._clean_text(html[:1000])}"  # Limit length for demo
            
            doc_id = document.get("document_url", "")
            passage_id = f"doc_{hashlib.md5(doc_id.encode()).hexdigest()[:8]}"
            
            passages.append((passage_id, passage_text))
        
        return passages
    
    def convert_record(self, source_record: Dict[str, Any]) -> List[Tuple[GoldenRecord, List[CorpusRecord]]]:
        """Convert NQ record to GoldenRecord and CorpusRecords
        
        Args:
            source_record: NQ record
            
        Returns:
            Tuple of (GoldenRecord, List[CorpusRecord]) or None if conversion fails
        """
        try:
            # Extract question
            question = source_record.get("question", "")
            question = self._clean_text(question)
            
            if not question:
                raise ValueError("question cannot be empty")
            
            # Extract answer from annotations
            annotations = source_record.get("annotations", [])
            if not annotations:
                raise ValueError("annotations cannot be empty")
            
            answer = self._extract_answer(annotations[0]) if annotations else ""
            if not answer:
                raise ValueError("answer cannot be empty")
            
            # Extract document and relevant passages
            document = source_record.get("document", {})
            if not document:
                raise ValueError("document cannot be empty")
            
            # Extract relevant passages
            passages = self._extract_relevant_passages(document, annotations[0])
            if not passages:
                raise ValueError("extract relevant passages failed")
            
            # Build reference contexts and corpus records
            reference_contexts = []
            reference_context_ids = []
            corpus_records = []
            
            for passage_id, passage_text in passages:
                reference_contexts.append(passage_text)
                reference_context_ids.append(passage_id)
                
                corpus_records.append(CorpusRecord(
                    reference_context=passage_text,
                    reference_context_id=passage_id,
                    title=document.get("title", "Unknown"),
                    metadata={
                        "source": "natural_questions",
                        "document_url": document.get("document_url", ""),
                        "relevance": "high"
                    }
                ))

            source_id = source_record.get("id", "")
            # Create metadata
            metadata = {
                "source": "natural_questions",
                "subset": self.subset,
                "source_id": source_id,
                "document_url": document.get("document_url", ""),
                "example_id": source_record.get("example_id", "")
            }
            
            # Create GoldenRecord with generated ID
            golden_id = source_id if source_id else self._generate_id()
            golden_record = GoldenRecord(
                id=golden_id,
                user_input=question,
                reference=answer,
                reference_contexts=reference_contexts,
                reference_context_ids=reference_context_ids,
                metadata=metadata
            )
            
            return [(golden_record, corpus_records)]
            
        except Exception as e:
            raise ValueError(f"Failed to convert NQ record: {e}")

    def create_metadata(self, source_path: Union[str, Path], num_records: int) -> DatasetMetadata:
        """Create metadata for the converted dataset
        
        Args:
            source_path: Path to source data
            num_records: Number of converted records
            
        Returns:
            DatasetMetadata object
        """
        return DatasetMetadata(
            name="natural_questions",
            version="1.0",
            description=f"Natural Questions dataset converted to Golden Dataset format ({self.subset} split)",
            source="https://ai.google.com/research/pubs/pub47761",
            size=num_records,
            domain="general_knowledge",
            language="en",
            created_at=None,
            license="CC BY-SA 3.0",
            tags=["search", "real_questions", "question_answering", "web"]
        )
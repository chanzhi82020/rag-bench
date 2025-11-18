"""XQuAD dataset converter to Golden Dataset format"""

import json
import logging
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)

from rag_benchmark.datasets.converters.base import BaseConverter
from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord, DatasetMetadata


class XQuADConverter(BaseConverter):
    """Converter for XQuAD dataset to Golden Dataset format"""
    
    def __init__(self, 
                 output_dir: Union[str, Path],
                 language: str = "zh",
                 batch_size: int = 1000):
        """Initialize XQuAD converter
        
        Args:
            output_dir: Output directory for converted dataset
            language: Language code (default: "zh" for Chinese)
            batch_size: Number of records to process in each batch
        """
        super().__init__(output_dir, batch_size)
        self.language = language
        self._paragraph_counter = 0
    
    def load_source_data(self, source_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Load XQuAD dataset from JSON file
        
        Args:
            source_path: Path to XQuAD JSON file
            
        Returns:
            Iterator of data elements containing paragraphs and title
        """
        source_path = Path(source_path)
        
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract data elements from XQuAD structure
        # XQuAD format: data -> [paragraphs, title]
        for article in data.get("data", []):
            yield article
    
    def convert_record(self, source_record: Dict[str, Any]) -> List[Tuple[GoldenRecord, List[CorpusRecord]]]:
        """Convert a XQuAD data element to multiple GoldenRecords and CorpusRecords
        
        Args:
            source_record: XQuAD data element containing paragraphs and title
            
        Returns:
            List of tuples (GoldenRecord, List[CorpusRecord]) for each Q&A pair
        """
        results = []
        
        try:
            title = source_record.get("title", "")
            paragraphs = source_record.get("paragraphs", [])
            
            if not paragraphs:
                raise ValueError("paragraphs cannot be empty")
            
            # Process each paragraph with unique index
            for para_index, paragraph in enumerate(paragraphs):
                context = paragraph.get("context", "")
                qas = paragraph.get("qas", [])
                
                if not context or not qas:
                    continue
                
                # Create globally unique corpus record ID
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
                
                # Process each Q&A pair in this paragraph
                for qa in qas:
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    qa_id = qa.get("id", "")
                    
                    if not question or not answers:
                        continue
                    
                    # Use the first answer (XQuAD may have multiple answers)
                    answer = answers[0].get("text", "")
                    answer_start = answers[0].get("answer_start", -1)
                    
                    if not answer:
                        continue
                    
                    # Create golden record
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
            
        except Exception as e:
            raise ValueError(f"Failed to convert XQuAD record: {e}")

    def create_metadata(self, source_path: Union[str, Path], num_records: int) -> DatasetMetadata:
        """Create metadata for the converted dataset
        
        Args:
            source_path: Path to source data
            num_records: Number of converted records
            
        Returns:
            DatasetMetadata object
        """
        return DatasetMetadata(
            name=f"xquad_{self.language}",
            version="1.0",
            description=f"XQuAD dataset ({self.language} language) converted to Golden Dataset format",
            source="https://github.com/google-deepmind/xquad",
            size=num_records,
            domain="question_answering",
            language=self.language,
            created_at=None,
            license="Apache 2.0",
            tags=["xquad", "question_answering", "multi_answer", self.language]
        )
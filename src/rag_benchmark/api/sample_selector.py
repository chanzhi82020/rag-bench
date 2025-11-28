"""Sample selection module for evaluation tasks"""

import random
from typing import List, Optional
import logging

from rag_benchmark.datasets.golden import GoldenDataset
from rag_benchmark.datasets.schemas.golden import GoldenRecord
from rag_benchmark.datasets.loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class SampleSelector:
    """Handles dataset sample selection strategies"""
    
    @staticmethod
    def select_by_ids(dataset: GoldenDataset, sample_ids: List[str]) -> GoldenDataset:
        """Select specific samples by ID
        
        Args:
            dataset: Source dataset
            sample_ids: List of record IDs
            
        Returns:
            New GoldenDataset with selected samples
            
        Raises:
            ValueError: If any sample ID is invalid
        """
        # Call dataset.select_by_ids() directly
        return dataset.select_by_ids(sample_ids)
    
    @staticmethod
    def select_random(dataset: GoldenDataset, n: int, seed: Optional[int] = None) -> GoldenDataset:
        """Select random samples from dataset
        
        Args:
            dataset: Source dataset
            n: Number of samples to select
            seed: Optional random seed for reproducibility
            
        Returns:
            New GoldenDataset with randomly selected samples
            
        Raises:
            ValueError: If n is negative or zero
        """
        if n <= 0:
            raise ValueError(f"Sample size must be positive, got {n}")
        
        # Use the existing create_subset method which handles random sampling
        return dataset.select_random(n, seed)
    
    @staticmethod
    def select_all(dataset: GoldenDataset) -> GoldenDataset:
        """Select all samples from dataset
        
        This returns the original dataset unchanged.
        
        Args:
            dataset: Source dataset
            
        Returns:
            The original dataset
        """
        return dataset
    
    @staticmethod
    def validate_sample_ids(dataset: GoldenDataset, sample_ids: List[str]) -> List[str]:
        """Validate that sample IDs exist in dataset
        
        Args:
            dataset: Dataset to validate against
            sample_ids: List of record IDs to validate
            
        Returns:
            List of invalid sample IDs (empty if all valid)
        """
        # Get valid IDs from dataset
        valid_ids = set(dataset.get_record_ids())
        
        # Check sample_ids against valid IDs
        invalid_ids = [sid for sid in sample_ids if sid not in valid_ids]
        
        return invalid_ids

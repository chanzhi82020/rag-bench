"""Checkpoint Manager for Evaluation Tasks

Handles checkpoint data persistence and loading for evaluation tasks,
enabling true checkpoint/resume capabilities.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ragas import EvaluationDataset

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint data for evaluation tasks
    
    This class handles:
    - Saving and loading checkpoint data for task stages
    - Persisting experiment datasets using pickle
    - Managing checkpoint file lifecycle
    """
    
    def __init__(self, tasks_dir: Path):
        """Initialize CheckpointManager
        
        Args:
            tasks_dir: Base directory for task data storage
        """
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_task_dir(self, task_id: str) -> Path:
        """Get the directory path for a task
        
        Args:
            task_id: Task identifier
            
        Returns:
            Path to task directory
        """
        return self.tasks_dir / task_id
    
    def save_checkpoint(self, task_id: str, stage: str, data: Dict[str, Any]) -> None:
        """Save checkpoint data for a stage
        
        Args:
            task_id: Task identifier
            stage: Stage name (e.g., 'load_dataset', 'prepare_experiment')
            data: Stage-specific data to save
            
        Raises:
            IOError: If checkpoint cannot be saved
        """
        try:
            task_dir = self._get_task_dir(task_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = task_dir / "checkpoint.json"
            
            # Load existing checkpoint or create new one
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
            else:
                checkpoint = {
                    "completed_stages": [],
                    "current_stage": None,
                    "stage_data": {},
                    "last_checkpoint_at": None
                }
            
            # Update checkpoint with new stage data
            if stage not in checkpoint["completed_stages"]:
                checkpoint["completed_stages"].append(stage)
            
            checkpoint["current_stage"] = stage
            checkpoint["stage_data"][stage] = {
                **data,
                "completed_at": datetime.now().isoformat()
            }
            checkpoint["last_checkpoint_at"] = datetime.now().isoformat()
            
            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info(f"Checkpoint saved for task '{task_id}', stage '{stage}'")
            
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint for task '{task_id}': {type(e).__name__}",
                exc_info=True,
                extra={
                    "task_id": task_id,
                    "operation": "save_checkpoint",
                    "stage": stage,
                    "error_type": type(e).__name__
                }
            )
            raise IOError(f"Failed to save checkpoint: {e}") from e
    
    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data if exists
        
        Args:
            task_id: Task identifier
            
        Returns:
            Checkpoint data dictionary or None if no checkpoint exists
            
        Note:
            Returns None if checkpoint file doesn't exist or is corrupted
        """
        try:
            task_dir = self._get_task_dir(task_id)
            checkpoint_file = task_dir / "checkpoint.json"
            
            if not checkpoint_file.exists():
                logger.debug(f"No checkpoint found for task '{task_id}'")
                return None
            
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            logger.info(
                f"Checkpoint loaded for task '{task_id}', "
                f"completed stages: {checkpoint.get('completed_stages', [])}"
            )
            return checkpoint
            
        except json.JSONDecodeError as e:
            logger.warning(
                f"Corrupted checkpoint file for task '{task_id}': {e}",
                extra={
                    "task_id": task_id,
                    "operation": "load_checkpoint",
                    "error_type": "JSONDecodeError"
                }
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to load checkpoint for task '{task_id}': {e}",
                exc_info=True,
                extra={
                    "task_id": task_id,
                    "operation": "load_checkpoint",
                    "error_type": type(e).__name__
                }
            )
            return None
    
    def save_golden_dataset(self, task_id: str, dataset: Any) -> None:
        """Save selected golden dataset using pickle
        
        Args:
            task_id: Task identifier
            dataset: Golden dataset object to serialize
            
        Raises:
            IOError: If dataset cannot be saved
        """
        try:
            task_dir = self._get_task_dir(task_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_file = task_dir / "golden_dataset.pkl"
            
            # Use pickle protocol 4 for better performance and compatibility
            with open(dataset_file, 'wb') as f:
                pickle.dump(dataset, f, protocol=4)
            
            # Get file size for logging
            file_size = dataset_file.stat().st_size
            logger.info(
                f"Golden dataset saved for task '{task_id}' "
                f"({file_size / 1024:.2f} KB)"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to save golden dataset for task '{task_id}': {type(e).__name__}",
                exc_info=True,
                extra={
                    "task_id": task_id,
                    "operation": "save_golden_dataset",
                    "error_type": type(e).__name__
                }
            )
            raise IOError(f"Failed to save golden dataset: {e}") from e
    
    def load_golden_dataset(self, task_id: str) -> Any:
        """Load selected golden dataset
        
        Args:
            task_id: Task identifier
            
        Returns:
            Deserialized golden dataset
            
        Raises:
            ValueError: If dataset file doesn't exist or is corrupted
        """
        try:
            task_dir = self._get_task_dir(task_id)
            dataset_file = task_dir / "golden_dataset.pkl"
            
            if not dataset_file.exists():
                raise ValueError(f"No golden dataset found for task '{task_id}'")
            
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
            
            logger.info(f"Golden dataset loaded for task '{task_id}'")
            return dataset
            
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Corrupted golden dataset for task '{task_id}': {e}")
        except Exception as e:
            raise ValueError(f"Failed to load golden dataset for task '{task_id}': {e}")
    
    def save_experiment_dataset(self, task_id: str, dataset: EvaluationDataset) -> None:
        """Save prepared experiment dataset using pickle
        
        Args:
            task_id: Task identifier
            dataset: Experiment dataset object to serialize
            
        Raises:
            IOError: If dataset cannot be saved
        """
        try:
            task_dir = self._get_task_dir(task_id)
            task_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_file = task_dir / "experiment_dataset.pkl"
            
            # Use pickle protocol 4 for better performance and compatibility
            with open(dataset_file, 'wb') as f:
                pickle.dump(dataset, f, protocol=4)
            
            # Get file size for logging
            file_size = dataset_file.stat().st_size
            logger.info(
                f"Experiment dataset saved for task '{task_id}' "
                f"({file_size / 1024 / 1024:.2f} MB)"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to save experiment dataset for task '{task_id}': {type(e).__name__}",
                exc_info=True,
                extra={
                    "task_id": task_id,
                    "operation": "save_experiment_dataset",
                    "error_type": type(e).__name__
                }
            )
            raise IOError(f"Failed to save experiment dataset: {e}") from e
    
    def load_experiment_dataset(self, task_id: str) -> EvaluationDataset:
        """Load prepared experiment dataset
        
        Args:
            task_id: Task identifier
            
        Returns:
            Deserialized experiment dataset or None if not found
            
        Note:
            Returns None if dataset file doesn't exist or is corrupted
        """
        try:
            task_dir = self._get_task_dir(task_id)
            dataset_file = task_dir / "experiment_dataset.pkl"
            
            if not dataset_file.exists():
                raise ValueError(f"No experiment dataset found for task '{task_id}'")
            
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
            
            logger.info(f"Experiment dataset loaded for task '{task_id}'")
            return dataset
            
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Corrupted experiment dataset for task '{task_id}': {e}")
        except Exception as e:
            raise ValueError(f"Failed to load experiment dataset for task '{task_id}': {e}")
    
    def clear_checkpoints(self, task_id: str) -> None:
        """Clear all checkpoint data for a task
        
        Args:
            task_id: Task identifier
            
        Note:
            This removes checkpoint.json, golden_dataset.pkl and experiment_dataset.pkl files
            but preserves status.json and config.json
        """
        try:
            task_dir = self._get_task_dir(task_id)
            
            if not task_dir.exists():
                logger.debug(f"No task directory found for '{task_id}'")
                return
            
            # Remove checkpoint file
            checkpoint_file = task_dir / "checkpoint.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"Checkpoint file removed for task '{task_id}'")
            
            # Remove golden dataset file
            golden_dataset_file = task_dir / "golden_dataset.pkl"
            if golden_dataset_file.exists():
                golden_dataset_file.unlink()
                logger.info(f"Golden dataset file removed for task '{task_id}'")
            
            # Remove experiment dataset file
            experiment_dataset_file = task_dir / "experiment_dataset.pkl"
            if experiment_dataset_file.exists():
                experiment_dataset_file.unlink()
                logger.info(f"Experiment dataset file removed for task '{task_id}'")
            
            logger.info(f"All checkpoint data cleared for task '{task_id}'")
            
        except Exception as e:
            logger.error(
                f"Failed to clear checkpoints for task '{task_id}': {e}",
                exc_info=True,
                extra={
                    "task_id": task_id,
                    "operation": "clear_checkpoints",
                    "error_type": type(e).__name__
                }
            )
            # Don't raise - clearing checkpoints is a cleanup operation

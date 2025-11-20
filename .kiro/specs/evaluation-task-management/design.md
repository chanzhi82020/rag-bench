# Design Document: Evaluation Task Management Improvements

## Overview

This design enhances the evaluation task management system to support task deletion, flexible sample selection strategies, and true checkpoint/resume capabilities through persistent intermediate data. The current system only persists task status but not intermediate results like prepared experiment datasets, making true resume functionality impossible.

The design introduces:
1. Task deletion API with cleanup of all associated files
2. Flexible sample selection (specific IDs, random sampling, or all samples)
3. Persistent checkpoint data including experiment datasets
4. Task retry functionality
5. Enhanced task status reporting with sample tracking

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│                                                              │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  Evaluation    │────────▶│  Task            │           │
│  │  Endpoints     │         │  Manager         │           │
│  └────────────────┘         └──────────────────┘           │
│         │                            │                      │
│         │                            │                      │
│         ▼                            ▼                      │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  Background    │         │  Checkpoint      │           │
│  │  Tasks         │         │  Manager         │           │
│  └────────────────┘         └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  data/tasks/     │
                            │  ├── task_id/    │
                            │  │   ├── status.json
                            │  │   ├── config.json
                            │  │   ├── experiment_dataset.pkl
                            │  │   ├── checkpoint.json
                            │  │   └── results.json
                            └──────────────────┘
```

### Component Interaction Flow

**Task Creation Flow:**
1. User calls `/evaluate/start` with sample selection strategy
2. System validates sample selection parameters
3. System creates task directory structure
4. System saves task configuration and initial status
5. Background task starts execution

**Task Execution with Checkpoints:**
1. Load checkpoint data if exists
2. Stage 1: Load dataset → Save checkpoint
3. Stage 2: Prepare experiment dataset → Save experiment dataset + checkpoint
4. Stage 3: Run evaluation → Save results + checkpoint
5. Stage 4: Process results → Save final results + update status

**Task Deletion Flow:**
1. User calls `/evaluate/delete/{task_id}`
2. System stops task if running
3. System removes task directory and all files
4. System removes task from memory

**Task Retry Flow:**
1. User calls `/evaluate/retry/{task_id}`
2. System loads original task configuration
3. System creates new task with same config
4. System starts new task execution

## Components and Interfaces

### 1. Task Manager Module

**Location:** `src/rag_benchmark/api/task_manager.py`

**Purpose:** Centralized task lifecycle management

**Key Functions:**

```python
class TaskManager:
    """Manages evaluation task lifecycle and persistence"""
    
    def create_task(self, config: EvaluateRequest) -> str:
        """Create a new evaluation task"""
        
    def delete_task(self, task_id: str) -> None:
        """Delete a task and all associated data"""
        
    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get current task status"""
        
    def list_tasks(self, filter_status: Optional[str] = None) -> List[TaskStatus]:
        """List all tasks with optional status filter"""
        
    def retry_task(self, task_id: str) -> str:
        """Create a new task with same config as failed task"""
        
    def stop_task(self, task_id: str) -> None:
        """Stop a running task"""
```

### 2. Checkpoint Manager Module

**Location:** `src/rag_benchmark/api/checkpoint_manager.py`

**Purpose:** Handles checkpoint data persistence and loading

**Key Functions:**

```python
class CheckpointManager:
    """Manages checkpoint data for evaluation tasks"""
    
    def save_checkpoint(self, task_id: str, stage: str, data: Dict) -> None:
        """Save checkpoint data for a stage"""
        
    def load_checkpoint(self, task_id: str) -> Optional[Dict]:
        """Load checkpoint data if exists"""
        
    def save_experiment_dataset(self, task_id: str, dataset: Any) -> None:
        """Save prepared experiment dataset"""
        
    def load_experiment_dataset(self, task_id: str) -> Optional[Any]:
        """Load prepared experiment dataset"""
        
    def clear_checkpoints(self, task_id: str) -> None:
        """Clear all checkpoint data for a task"""
```

### 3. Sample Selector Module

**Location:** `src/rag_benchmark/api/sample_selector.py`

**Purpose:** Handles sample selection strategies

**Key Functions:**

```python
class SampleSelector:
    """Handles dataset sample selection"""
    
    @staticmethod
    def select_by_ids(dataset: GoldenDataset, sample_ids: List[str]) -> GoldenDataset:
        """Select specific samples by ID"""
        
    @staticmethod
    def select_random(dataset: GoldenDataset, n: int, seed: Optional[int] = None) -> GoldenDataset:
        """Select random samples"""
        
    @staticmethod
    def select_all(dataset: GoldenDataset) -> GoldenDataset:
        """Select all samples"""
        
    @staticmethod
    def validate_sample_ids(dataset: GoldenDataset, sample_ids: List[str]) -> List[str]:
        """Validate that sample IDs exist in dataset"""
```

### 4. Enhanced API Endpoints

**Modified Endpoints:**

- `POST /evaluate/start` - Add sample selection parameters
- `GET /evaluate/status/{task_id}` - Include sample tracking info
- `GET /evaluate/tasks` - Add filtering by status

**New Endpoints:**

- `DELETE /evaluate/delete/{task_id}` - Delete evaluation task
- `POST /evaluate/retry/{task_id}` - Retry failed task
- `POST /evaluate/stop/{task_id}` - Stop running task
- `GET /evaluate/{task_id}/samples` - Get sample details for task

## Data Models

### Directory Structure

```
data/
├── tasks/
│   ├── task_id_1/
│   │   ├── status.json              # Task status
│   │   ├── config.json              # Task configuration
│   │   ├── experiment_dataset.pkl   # Prepared experiment dataset
│   │   ├── checkpoint.json          # Checkpoint data
│   │   └── results.json             # Final results
│   └── task_id_2/
│       ├── status.json
│       ├── config.json
│       ├── experiment_dataset.pkl
│       ├── checkpoint.json
│       └── results.json
```

### File Formats

**1. status.json**
```json
{
  "task_id": "uuid",
  "status": "running",
  "progress": 0.5,
  "current_stage": "Running evaluation",
  "result": null,
  "error": null,
  "created_at": "2025-11-24T10:00:00",
  "updated_at": "2025-11-24T10:05:00",
  "sample_info": {
    "selection_strategy": "specific_ids",
    "total_samples": 100,
    "selected_samples": 10,
    "sample_ids": ["id1", "id2", "..."],
    "completed_samples": 5,
    "failed_samples": 0
  }
}
```

**2. config.json**
```json
{
  "dataset_name": "hotpotqa",
  "subset": null,
  "rag_name": "baseline",
  "eval_type": "e2e",
  "model_info": {
    "llm_model_id": "gpt-4",
    "embedding_model_id": "text-embedding-3-small"
  },
  "sample_selection": {
    "strategy": "specific_ids",
    "sample_ids": ["id1", "id2", "..."],
    "sample_size": null,
    "random_seed": null
  }
}
```

**3. checkpoint.json**
```json
{
  "completed_stages": ["load_dataset", "prepare_experiment"],
  "current_stage": "run_evaluation",
  "stage_data": {
    "load_dataset": {
      "completed_at": "2025-11-24T10:01:00",
      "dataset_size": 100
    },
    "prepare_experiment": {
      "completed_at": "2025-11-24T10:03:00",
      "experiment_dataset_path": "experiment_dataset.pkl"
    }
  },
  "last_checkpoint_at": "2025-11-24T10:03:00"
}
```

**4. experiment_dataset.pkl**
- Format: Python pickle format
- Content: Serialized EvaluationDataset object
- Created by: `pickle.dump()`
- Loaded by: `pickle.load()`

**5. results.json**
```json
{
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78
  },
  "detailed_results": [
    {
      "user_input": "query",
      "response": "answer",
      "faithfulness": 0.9,
      "answer_relevancy": 0.8
    }
  ],
  "sample_count": 10,
  "eval_type": "e2e",
  "completed_at": "2025-11-24T10:10:00"
}
```

### API Request/Response Models

**SampleSelection (new Pydantic model):**
```python
class SampleSelection(BaseModel):
    strategy: str  # "specific_ids", "random", "all"
    sample_ids: Optional[List[str]] = None
    sample_size: Optional[int] = None
    random_seed: Optional[int] = None
```

**Updated EvaluateRequest:**
```python
class EvaluateRequest(BaseModel):
    dataset_name: str
    subset: Optional[str] = None
    rag_name: str
    eval_type: str = "e2e"
    model_info: ModelConfig
    sample_selection: Optional[SampleSelection] = None  # NEW
```

**SampleInfo (new Pydantic model):**
```python
class SampleInfo(BaseModel):
    selection_strategy: str
    total_samples: int
    selected_samples: int
    sample_ids: Optional[List[str]] = None
    completed_samples: int
    failed_samples: int
```

**Updated TaskStatus:**
```python
class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    current_stage: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    sample_info: Optional[SampleInfo] = None  # NEW
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Task deletion completeness

*For any* evaluation task, deleting the task should remove all associated files (status.json, config.json, experiment_dataset.pkl, checkpoint.json, results.json) and the task directory itself.

**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Sample selection validation

*For any* evaluation request with specific sample IDs, all provided sample IDs should exist in the dataset, otherwise the system should reject the request with a clear error message.

**Validates: Requirements 2.4**

### Property 3: Checkpoint resume consistency

*For any* evaluation task that is interrupted and resumed, the resumed task should skip already completed stages and continue from the last checkpoint, producing the same final result as if it had run without interruption.

**Validates: Requirements 3.2, 3.3, 3.4**

### Property 4: Sample tracking accuracy

*For any* completed evaluation task, the reported sample information (total_samples, selected_samples, completed_samples, failed_samples) should accurately reflect the actual samples processed.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

### Property 5: Task retry configuration preservation

*For any* failed evaluation task, retrying the task should create a new task with identical configuration (dataset, RAG, eval_type, model_info, sample_selection) as the original task.

**Validates: Requirements 6.2**

### Property 6: Checkpoint data integrity

*For any* evaluation task with saved checkpoint data, loading the checkpoint should restore the exact state of completed stages, allowing the task to continue without re-executing completed work.

**Validates: Requirements 3.1, 3.2, 3.3**

### Property 7: Task directory isolation

*For any* set of evaluation tasks, each task's data should be stored in its own isolated directory, and operations on one task should not affect other tasks' data.

**Validates: Requirements 5.1**

### Property 8: Sample selection strategy consistency

*For any* evaluation task, the sample selection strategy used should match the strategy specified in the request, and the actual samples evaluated should match the selection criteria.

**Validates: Requirements 2.1, 2.2, 2.3, 2.5**

## Error Handling

### Error Scenarios and Handling

**1. Task Not Found**
- **Scenario:** User tries to delete/retry/stop a non-existent task
- **Handling:**
  - Return 404 HTTP error
  - Provide clear error message with task ID
  - Suggest listing available tasks

**2. Invalid Sample IDs**
- **Scenario:** User provides sample IDs that don't exist in dataset
- **Handling:**
  - Validate sample IDs before starting task
  - Return 400 HTTP error with list of invalid IDs
  - Suggest valid sample ID format

**3. Corrupted Checkpoint Data**
- **Scenario:** Checkpoint file is corrupted or incompatible
- **Handling:**
  - Log warning with task ID
  - Clear corrupted checkpoint
  - Restart evaluation from beginning
  - Update task status to indicate restart

**4. Disk Space Exhaustion**
- **Scenario:** Insufficient disk space to save checkpoint/results
- **Handling:**
  - Log error with disk space info
  - Mark task as failed
  - Provide actionable error message to user
  - Keep task data for retry

**5. Task Already Running**
- **Scenario:** User tries to retry a task that's still running
- **Handling:**
  - Return 409 HTTP error (Conflict)
  - Provide current task status
  - Suggest stopping task first

**6. Experiment Dataset Serialization Failure**
- **Scenario:** Cannot pickle experiment dataset
- **Handling:**
  - Log error with exception details
  - Continue task execution without checkpoint
  - Warn user that resume won't be available
  - Complete task normally

### Error Logging Format

```python
logger.error(
    f"Failed to save checkpoint for task '{task_id}': {error_type}",
    exc_info=True,
    extra={
        "task_id": task_id,
        "operation": "save_checkpoint",
        "stage": stage,
        "error_type": type(e).__name__
    }
)
```

### User-Facing Error Messages

```python
# Invalid sample IDs
{
  "error": "Invalid sample IDs provided",
  "details": "The following sample IDs do not exist in the dataset: ['id1', 'id2']",
  "action": "Please verify sample IDs and try again"
}

# Task not found
{
  "error": "Evaluation task not found",
  "details": "Task ID 'abc-123' does not exist",
  "action": "Use GET /evaluate/tasks to list available tasks"
}

# Disk space error
{
  "error": "Insufficient disk space to save checkpoint",
  "details": "Required: 500MB, Available: 100MB",
  "action": "Free up disk space and retry the task"
}
```

## Testing Strategy

### Unit Tests

**Test Coverage:**

1. **Task Manager**
   - Test task creation with different sample selection strategies
   - Test task deletion removes all files
   - Test task listing with status filters
   - Test task retry creates new task with same config

2. **Checkpoint Manager**
   - Test checkpoint save and load
   - Test experiment dataset serialization
   - Test checkpoint clear
   - Test handling of corrupted checkpoints

3. **Sample Selector**
   - Test selection by specific IDs
   - Test random selection with seed
   - Test selection of all samples
   - Test validation of sample IDs

4. **API Endpoints**
   - Test `/evaluate/start` with sample selection
   - Test `/evaluate/delete/{task_id}` removes task
   - Test `/evaluate/retry/{task_id}` creates new task
   - Test `/evaluate/stop/{task_id}` stops running task

### Property-Based Tests

**Testing Framework:** Use `hypothesis` library for Python property-based testing

**Test Configuration:** Each property test should run minimum 100 iterations

**Property Tests:**

1. **Task deletion completeness** (Property 1)
   - Generate random tasks
   - Delete tasks
   - Verify all files removed

2. **Sample selection validation** (Property 2)
   - Generate random sample IDs (valid and invalid)
   - Verify validation catches invalid IDs

3. **Checkpoint resume consistency** (Property 3)
   - Generate random evaluation tasks
   - Interrupt at random stages
   - Resume and verify same results

4. **Sample tracking accuracy** (Property 4)
   - Generate random evaluation tasks
   - Verify sample counts match actual processing

5. **Task retry configuration preservation** (Property 5)
   - Generate random task configs
   - Retry tasks
   - Verify configs match

6. **Checkpoint data integrity** (Property 6)
   - Generate random checkpoint data
   - Save and load
   - Verify data matches

7. **Task directory isolation** (Property 7)
   - Create multiple tasks
   - Verify each has isolated directory

8. **Sample selection strategy consistency** (Property 8)
   - Generate random selection strategies
   - Verify actual samples match strategy

### Integration Tests

1. **End-to-end task lifecycle**
   - Create task → Execute → Complete → Delete
   - Verify all stages work correctly

2. **Checkpoint resume**
   - Start task → Interrupt → Resume
   - Verify results are consistent

3. **Concurrent task execution**
   - Multiple tasks running simultaneously
   - Verify no data corruption

4. **Large dataset handling**
   - Evaluate with 1000+ samples
   - Verify checkpoint and resume work

## Implementation Notes

### Performance Considerations

1. **Checkpoint Frequency:** Save checkpoints after each major stage, not after each sample
2. **Pickle Optimization:** Use protocol 4 or 5 for faster serialization
3. **Async I/O:** Use async file operations for checkpoint saves
4. **Memory Management:** Clear experiment dataset from memory after saving to disk

### Security Considerations

1. **Path Traversal:** Validate task IDs to prevent directory traversal attacks
2. **File Permissions:** Set appropriate permissions on task directories (0755)
3. **Disk Quotas:** Consider implementing per-task disk usage limits
4. **Input Validation:** Validate all sample selection parameters

### Scalability Considerations

1. **Task Limit:** Current design suitable for <10,000 tasks
2. **Large Checkpoints:** Experiment datasets >1GB may have slow save/load times
3. **Future Enhancement:** Consider database storage for task metadata

### Backward Compatibility

**Migration Strategy:**
- Existing tasks in old format (single JSON file) will continue to work
- New tasks will use new directory structure
- Provide migration script to convert old tasks to new format
- Old tasks won't have checkpoint/resume capability

## Dependencies

### New Dependencies

- No new external dependencies needed
- Uses existing: `pickle`, `json`, `pathlib`

### Modified Files

1. `src/rag_benchmark/api/main.py`
   - Add new endpoints
   - Update evaluation task execution
   - Add checkpoint logic

2. New file: `src/rag_benchmark/api/task_manager.py`
   - Implement TaskManager class

3. New file: `src/rag_benchmark/api/checkpoint_manager.py`
   - Implement CheckpointManager class

4. New file: `src/rag_benchmark/api/sample_selector.py`
   - Implement SampleSelector class

5. `src/rag_benchmark/datasets/golden.py`
   - Add method to select samples by ID

## Deployment Considerations

### Migration Steps

1. Deploy new code
2. Run migration script to convert existing tasks
3. Test checkpoint/resume functionality
4. Update frontend to use new endpoints

### Rollback Plan

If issues arise:
1. Revert to previous code version
2. New task directories remain on disk (harmless)
3. Old tasks continue to work
4. New features unavailable until fixed

### Monitoring

Add metrics for:
- Task creation/deletion rate
- Checkpoint save/load success rate
- Average task execution time
- Disk space usage for task data
- Task retry rate

# Implementation Plan

- [ ] 1. Create task directory structure and update task persistence
  - Update task creation to use directory structure (task_id/status.json, config.json)
  - Migrate existing single-file tasks to new directory structure
  - Update `save_task_status()` to save to task directory
  - Update `load_task_status()` to load from task directory
  - _Requirements: 5.1, 5.2, 5.5_

- [ ]* 1.1 Write property test for task directory isolation
  - **Property 7: Task directory isolation**
  - **Validates: Requirements 5.1**

- [ ] 2. Implement CheckpointManager module
  - Create `src/rag_benchmark/api/checkpoint_manager.py`
  - Implement `save_checkpoint()` method
  - Implement `load_checkpoint()` method
  - Implement `save_experiment_dataset()` method using pickle
  - Implement `load_experiment_dataset()` method
  - Implement `clear_checkpoints()` method
  - _Requirements: 3.1, 3.2, 3.3, 5.3, 5.4_

- [ ]* 2.1 Write property test for checkpoint data integrity
  - **Property 6: Checkpoint data integrity**
  - **Validates: Requirements 3.1, 3.2, 3.3**

- [ ] 3. Implement SampleSelector module
  - Create `src/rag_benchmark/api/sample_selector.py`
  - Implement `select_by_ids()` method
  - Implement `select_random()` method
  - Implement `select_all()` method
  - Implement `validate_sample_ids()` method
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 3.1 Write property test for sample selection validation
  - **Property 2: Sample selection validation**
  - **Validates: Requirements 2.4**

- [ ]* 3.2 Write property test for sample selection strategy consistency
  - **Property 8: Sample selection strategy consistency**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**

- [ ] 4. Add sample selection to GoldenDataset
  - Add `select_by_ids()` method to GoldenDataset class
  - Update method to return new GoldenDataset with filtered records
  - Ensure corpus records are also filtered appropriately
  - _Requirements: 2.1, 2.4_

- [ ] 5. Update EvaluateRequest model with sample selection
  - Create `SampleSelection` Pydantic model
  - Add `sample_selection` field to `EvaluateRequest`
  - Update API documentation
  - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [ ] 6. Update TaskStatus model with sample tracking
  - Create `SampleInfo` Pydantic model
  - Add `sample_info` field to `TaskStatus`
  - Update task status serialization/deserialization
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ]* 6.1 Write property test for sample tracking accuracy
  - **Property 4: Sample tracking accuracy**
  - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [ ] 7. Implement checkpoint/resume in evaluation task execution
  - Update `run_evaluation_task()` to check for existing checkpoint
  - Add checkpoint saves after each stage (load dataset, prepare experiment, run evaluation)
  - Implement stage skipping logic based on checkpoint
  - Add experiment dataset persistence after preparation
  - Handle corrupted checkpoint data gracefully
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ]* 7.1 Write property test for checkpoint resume consistency
  - **Property 3: Checkpoint resume consistency**
  - **Validates: Requirements 3.2, 3.3, 3.4**

- [ ] 8. Implement sample selection in evaluation task execution
  - Update `run_evaluation_task()` to apply sample selection strategy
  - Use `SampleSelector` to filter dataset based on request
  - Track selected sample IDs in task status
  - Update sample info throughout execution
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3_

- [ ] 9. Implement TaskManager module
  - Create `src/rag_benchmark/api/task_manager.py`
  - Implement `create_task()` method
  - Implement `delete_task()` method with file cleanup
  - Implement `get_task_status()` method
  - Implement `list_tasks()` method with status filtering
  - Implement `retry_task()` method
  - Implement `stop_task()` method
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 6.1, 6.2, 6.3, 6.4_

- [ ]* 9.1 Write property test for task deletion completeness
  - **Property 1: Task deletion completeness**
  - **Validates: Requirements 1.1, 1.2, 1.3**

- [ ]* 9.2 Write property test for task retry configuration preservation
  - **Property 5: Task retry configuration preservation**
  - **Validates: Requirements 6.2**

- [ ] 10. Add DELETE /evaluate/delete/{task_id} endpoint
  - Implement endpoint to delete evaluation task
  - Stop task if running before deletion
  - Use TaskManager.delete_task()
  - Return success confirmation
  - Add error handling for non-existent tasks
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 11. Add POST /evaluate/retry/{task_id} endpoint
  - Implement endpoint to retry failed task
  - Load original task configuration
  - Create new task with same config
  - Use TaskManager.retry_task()
  - Return new task ID
  - Add error handling for non-existent or running tasks
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 12. Add POST /evaluate/stop/{task_id} endpoint
  - Implement endpoint to stop running task
  - Use TaskManager.stop_task()
  - Update task status to "stopped"
  - Return success confirmation
  - Add error handling for non-existent or already completed tasks
  - _Requirements: 1.4_

- [ ] 13. Add GET /evaluate/{task_id}/samples endpoint
  - Implement endpoint to get sample details for task
  - Return sample selection strategy, sample IDs, and per-sample results
  - Add error handling for non-existent tasks
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 14. Update POST /evaluate/start endpoint
  - Add sample_selection parameter handling
  - Validate sample selection parameters
  - Save sample selection config to task config.json
  - Initialize sample_info in task status
  - Add error handling for invalid sample IDs
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 15. Update GET /evaluate/status/{task_id} endpoint
  - Include sample_info in response
  - Add checkpoint progress information
  - Update response model documentation
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 16. Update GET /evaluate/tasks endpoint
  - Add status filter parameter (pending, running, completed, failed)
  - Update to load tasks from new directory structure
  - Include sample_info in task list
  - _Requirements: 4.1, 4.2_

- [ ] 17. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 18. Add comprehensive error messages
  - Implement user-friendly error messages for invalid sample IDs
  - Add error messages for task not found scenarios
  - Add error messages for disk space issues
  - Add error messages for corrupted checkpoint data
  - _Requirements: 2.4, 3.5_

- [ ]* 18.1 Write unit tests for error handling
  - Test invalid sample ID error messages
  - Test task not found error messages
  - Test disk space error messages
  - Test corrupted checkpoint error messages

- [ ] 19. Create migration script for existing tasks
  - Create script to convert old single-file tasks to new directory structure
  - Test migration with existing task files
  - Document migration process
  - _Requirements: 5.1, 5.2_

- [ ] 20. Update frontend to use new endpoints
  - Add delete button for evaluation tasks
  - Add retry button for failed tasks
  - Add stop button for running tasks
  - Display sample selection info in task details
  - Display checkpoint progress
  - _Requirements: 1.5, 4.1, 4.2, 6.1_

- [ ] 21. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

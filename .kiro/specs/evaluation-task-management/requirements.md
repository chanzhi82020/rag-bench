# Requirements Document

## Introduction

当前评测任务系统存在以下问题：
1. 评测任务无法删除，导致任务列表不断增长
2. 评测时只能使用随机采样，无法选择特定的数据集样本进行评测
3. 评测任务的中间数据（如准备好的实验数据集）没有持久化，导致断点续传功能无法真正实现

本需求旨在改进评测任务管理系统，使其支持任务删除、灵活的样本选择策略，以及真正的断点续传能力。

## Glossary

- **Evaluation Task**: 评测任务，包含数据集、RAG实例、评测类型等配置的异步评测作业
- **Task Status**: 任务状态，包括pending、running、completed、failed等状态
- **Experiment Dataset**: 实验数据集，包含原始数据和RAG系统生成的检索结果、答案等
- **Sample Selection**: 样本选择，指定评测使用哪些数据集样本的策略
- **Checkpoint Data**: 检查点数据，评测过程中的中间结果，用于断点续传
- **Task Persistence**: 任务持久化，将任务状态和中间数据保存到磁盘

## Requirements

### Requirement 1

**User Story:** 作为用户，我希望能够删除不需要的评测任务，以保持任务列表清洁

#### Acceptance Criteria

1. WHEN a user requests to delete an evaluation task THEN the system SHALL remove the task from the task list
2. WHEN a user deletes an evaluation task THEN the system SHALL remove the task status file from disk
3. WHEN a user deletes an evaluation task THEN the system SHALL remove all associated checkpoint data files
4. WHEN a user deletes a running evaluation task THEN the system SHALL stop the task execution before deletion
5. WHEN a user deletes an evaluation task THEN the system SHALL return a success confirmation

### Requirement 2

**User Story:** 作为用户，我希望能够选择特定的数据集样本进行评测，而不仅仅是随机采样

#### Acceptance Criteria

1. WHEN a user starts an evaluation THEN the system SHALL support specifying a list of sample IDs to evaluate
2. WHEN a user starts an evaluation THEN the system SHALL support specifying a random sample size
3. WHEN a user starts an evaluation THEN the system SHALL support evaluating all samples in the dataset
4. WHEN a user specifies sample IDs THEN the system SHALL validate that all IDs exist in the dataset
5. WHEN a user specifies both sample IDs and sample size THEN the system SHALL prioritize sample IDs

### Requirement 3

**User Story:** 作为用户，我希望评测任务能够真正支持断点续传，即使系统重启也能从中断处继续

#### Acceptance Criteria

1. WHEN an evaluation task prepares the experiment dataset THEN the system SHALL persist the experiment dataset to disk
2. WHEN an evaluation task completes a stage THEN the system SHALL update the checkpoint data on disk
3. WHEN an evaluation task is resumed THEN the system SHALL load the checkpoint data from disk
4. WHEN an evaluation task is resumed THEN the system SHALL skip already completed stages
5. WHEN checkpoint data is corrupted THEN the system SHALL restart the evaluation from the beginning

### Requirement 4

**User Story:** 作为用户，我希望能够查看评测任务使用了哪些样本，以便理解评测结果

#### Acceptance Criteria

1. WHEN a user views an evaluation task THEN the system SHALL display the sample selection strategy used
2. WHEN a user views an evaluation task THEN the system SHALL display the number of samples evaluated
3. WHEN a user views a completed evaluation task THEN the system SHALL display the list of sample IDs evaluated
4. WHEN a user views an evaluation task THEN the system SHALL display which samples succeeded and which failed
5. WHEN a user views an evaluation task THEN the system SHALL allow downloading the detailed sample results

### Requirement 5

**User Story:** 作为系统管理员，我希望评测任务的持久化数据有清晰的组织结构，便于管理和调试

#### Acceptance Criteria

1. WHEN the system persists evaluation task data THEN the system SHALL organize files in a dedicated directory per task
2. WHEN the system persists evaluation task data THEN the system SHALL store task status in a JSON file
3. WHEN the system persists evaluation task data THEN the system SHALL store experiment dataset in a separate file
4. WHEN the system persists evaluation task data THEN the system SHALL store checkpoint data in a separate file
5. WHEN the system persists evaluation task data THEN the system SHALL include metadata with timestamps and task configuration

### Requirement 6

**User Story:** 作为用户，我希望能够重新运行失败的评测任务，而不需要重新配置所有参数

#### Acceptance Criteria

1. WHEN a user views a failed evaluation task THEN the system SHALL provide a retry option
2. WHEN a user retries a failed evaluation task THEN the system SHALL use the same configuration as the original task
3. WHEN a user retries a failed evaluation task THEN the system SHALL create a new task ID
4. WHEN a user retries a failed evaluation task THEN the system SHALL preserve the original task for reference
5. WHEN a user retries a failed evaluation task THEN the system SHALL start from the beginning unless checkpoint data is valid

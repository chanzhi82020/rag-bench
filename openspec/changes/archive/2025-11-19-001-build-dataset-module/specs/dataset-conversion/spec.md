# Dataset Conversion

## ADDED Requirements

### Requirement: The system SHALL provide HotpotQA to Golden Dataset conversion
The converter SHALL extract question, answer and supporting facts, convert them to user_input, reference and reference_contexts, and generate corresponding corpus files, supporting both distractor and fullwiki versions.

#### Scenario:
Developers need to convert HotpotQA dataset to standard Golden Dataset format.

### Requirement: The system SHALL provide Natural Questions to Golden Dataset conversion
The converter SHALL process NQ's special annotation format, extract long and short answers, find related document fragments as reference_contexts, and handle annotation tags and HTML tags.

#### Scenario:
Users want to use Google's Natural Questions dataset for evaluation.

### Requirement: The system SHALL support custom dataset conversion templates
The system SHALL provide conversion templates and tools to guide users on mapping custom data to standard format, supporting multiple input formats like CSV, JSON, Excel.

#### Scenario:
Enterprises have their own intelligent customer service Q&A data that needs conversion.

### Requirement: The system SHALL support batch conversion with progress tracking
The system SHALL support batch conversion, provide progress bar display, record failed entries and reasons, support resume and parallel conversion.

#### Scenario:
When converting large datasets, users need to see conversion progress and error statistics.

### Requirement: The system SHALL validate converted data integrity
The system SHALL automatically validate conversion result integrity, ensure all records are correctly converted, reference_contexts can be found in corpus, data format meets standards, and generate integrity reports.

#### Scenario:
After data conversion, need to ensure data quality.

## MODIFIED Requirements

### Requirement: [base] Data conversion MUST preserve original data metadata
During conversion, original data metadata (such as data source, creation time, version, license, etc.) MUST be preserved for data lineage and version management tracking, ensuring data traceability.

#### Scenario:
Track data lineage and version management.

## REMOVED Requirements

[None]
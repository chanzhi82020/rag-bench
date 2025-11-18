# Golden Dataset Management

## ADDED Requirements

### Requirement: The system SHALL define and standardize Golden Dataset format
The system SHALL provide clear data format definitions and validation mechanisms to ensure data consistency and quality. The data format SHALL include user_input (user question), reference (reference answer), reference_contexts (related contexts) and other core fields.

#### Scenario:
Developers need to create a standards-compliant Golden Dataset for RAG evaluation.

### Requirement: The system SHALL support loading Golden Dataset by name
The system SHALL support loading data through dataset names, returning standardized GoldenRecord object lists, supporting both streaming and batch loading modes.

#### Scenario:
Users want to use the built-in HotpotQA dataset for evaluation.

### Requirement: The system SHALL provide Golden Dataset quality validation
The system SHALL automatically validate data quality, check required fields, data length reasonableness, context coverage, and provide detailed validation reports and repair suggestions.

#### Scenario:
When importing custom Golden Dataset, data quality needs to meet standards.

### Requirement: The system SHALL support multiple dataset registration and management
Users SHALL be able to view all available dataset lists, obtain dataset metadata (scale, domain, quality metrics, etc.), and register new datasets in the system.

#### Scenario:
The system needs to manage multiple built-in and custom datasets.

### Requirement: The system SHALL support efficient streaming for large datasets
The system SHALL support streaming loading to avoid memory overflow, while providing pagination and sampling functions for development debugging, supporting lazy loading and cache optimization.

#### Scenario:
Processing large datasets containing 100k records.

## MODIFIED Requirements

### Requirement: [base] All RAG evaluation data MUST follow unified data format
All RAG evaluation data MUST follow unified JSONL format, including user_input, reference, reference_contexts and other core fields, ensuring evaluation process consistency and result comparability.

#### Scenario:
Ensure evaluation process consistency and result comparability.

## REMOVED Requirements

[None]
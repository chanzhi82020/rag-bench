#!/usr/bin/env python3
"""
Convert datasets to Golden Dataset format
Supports: XQuAD, HotpotQA, Natural Questions
"""

import argparse
import sys
from pathlib import Path
from typing import Union

from scripts.converters import (
    convert_dataset,
    list_available_converters
)
from rag_benchmark.datasets.loaders.base import ValidationResult
from rag_benchmark.datasets.registry import list_datasets, create_custom_loader
from rag_benchmark.datasets.validators.format import FormatValidator
from rag_benchmark.datasets.validators.quality import QualityValidator


def validate_dataset(dataset_path: Union[str, Path]) -> ValidationResult:
    """Validate a dataset at the given path

    Args:
        dataset_path: Path to dataset directory

    Returns:
        ValidationResult with validation details
    """
    dataset_path = Path(dataset_path)

    loader = create_custom_loader(dataset_path)

    # Format validation
    format_errors = FormatValidator.validate_file_structure(dataset_path)
    if format_errors:
        result = ValidationResult()
        result.is_valid = False
        result.errors.extend(format_errors)
        return result

    # Load records for validation
    golden_records = list(loader.load_golden_records())
    corpus_records = list(loader.load_corpus_records())

    # Combine format and quality validation
    format_result = loader.validate()
    quality_result = QualityValidator.validate_dataset_quality(
        golden_records, corpus_records
    )

    # Merge results
    result = ValidationResult()
    result.is_valid = format_result.is_valid and quality_result.is_valid
    result.errors.extend(format_result.errors)
    result.errors.extend(quality_result.errors)
    result.warnings.extend(format_result.warnings)
    result.warnings.extend(quality_result.warnings)
    result.statistics.update(format_result.statistics)
    result.statistics.update(quality_result.statistics)

    return result



def main():
    parser = argparse.ArgumentParser(description="Convert datasets to Golden Dataset format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset file or HuggingFace dataset identifier"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=["xquad", "hotpotqa", "nq"],
        help="Type of dataset to convert"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Language code (default: zh)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Dataset variant (e.g., 'distractor' or 'fullwiki' for HotpotQA)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        help="Dataset subset (e.g., 'validation' or 'train' for Natural Questions)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the converted dataset"
    )

    args = parser.parse_args()
    
    # List available datasets
    print("Available datasets in registry:")
    for dataset in list_datasets():
        print(f"  - {dataset['name']}: {dataset['description']}")
    print()
    
    # List available converters
    print("Available converters:")
    converters = list_available_converters()
    for name, desc in converters.items():
        print(f"  - {name}: {desc}")
    print()

    # Build converter arguments
    converter_kwargs = {
        "batch_size": args.batch_size
    }
    
    # Add dataset-specific arguments
    if args.dataset_type == "xquad":
        converter_kwargs["language"] = args.language
        print(f"Converting XQuAD dataset from {args.input}...")
        print(f"Language: {args.language}")
        
    elif args.dataset_type == "hotpotqa":
        converter_kwargs["variant"] = args.variant or "distractor"
        print(f"Converting HotpotQA dataset from {args.input}...")
        print(f"Variant: {converter_kwargs['variant']}")
        
    elif args.dataset_type == "nq":
        converter_kwargs["subset"] = args.subset or "validation"
        print(f"Converting Natural Questions dataset from {args.input}...")
        print(f"Subset: {converter_kwargs['subset']}")
    
    print(f"Output directory: {args.output}")
    print()
    
    try:
        # Perform conversion
        result = convert_dataset(
            source_path=args.input,
            output_dir=args.output,
            converter_type=args.dataset_type,
            **converter_kwargs
        )
        
        # Print conversion results
        print("Conversion completed successfully!")
        print(f"  Golden records: {result.converted_records}")
        print(f"  Failed records: {result.failed_records}")
        print(f"  Output directory: {args.output}")
        
        if result.statistics:
            print("  Statistics:")
            for key, value in result.statistics.items():
                print(f"    - {key}: {value}")
        
        # Validate if requested
        if args.validate:
            print("\nValidating converted dataset...")
            try:
                validation_result = validate_dataset(args.output)
                
                if validation_result.is_valid:
                    print("✓ Dataset is valid!")
                else:
                    print("✗ Dataset validation failed:")
                    for error in validation_result.errors:
                        print(f"  - {error}")
                
                if validation_result.warnings:
                    print("\nWarnings:")
                    for warning in validation_result.warnings:
                        print(f"  - {warning}")
                        
            except Exception as e:
                print(f"Validation error: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
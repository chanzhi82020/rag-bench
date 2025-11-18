"""Converters module initialization and main API functions"""

from typing import Union, Dict, Any, Optional
from pathlib import Path

from rag_benchmark.datasets.converters.hotpotqa import HotpotQAConverter
from rag_benchmark.datasets.converters.nq import NaturalQuestionsConverter
from rag_benchmark.datasets.converters.xquad import XQuADConverter
from rag_benchmark.datasets.converters.base import BaseConverter, ConversionResult

# Registry of available converters
CONVERTER_REGISTRY = {
    "hotpotqa": HotpotQAConverter,
    "natural_questions": NaturalQuestionsConverter,
    "xquad": XQuADConverter,
}


def convert_dataset(
    source_path: Union[str, Path],
    output_dir: Union[str, Path],
    converter_type: str,
    **kwargs
) -> ConversionResult:
    """Convert a dataset to Golden Dataset format
    
    Args:
        source_path: Path to source dataset
        output_dir: Output directory for converted dataset
        converter_type: Type of converter to use
        **kwargs: Additional arguments for the converter
        
    Returns:
        ConversionResult with conversion details
    """
    if converter_type not in CONVERTER_REGISTRY:
        raise ValueError(f"Unknown converter type: {converter_type}")
    
    converter_class = CONVERTER_REGISTRY[converter_type]
    converter = converter_class(output_dir, **kwargs)
    
    return converter.convert(source_path)


def create_hotpotqa_converter(
    output_dir: Union[str, Path],
    variant: str = "distractor",
    **kwargs
) -> HotpotQAConverter:
    """Create a HotpotQA converter
    
    Args:
        output_dir: Output directory
        variant: HotpotQA variant ("distractor" or "fullwiki")
        **kwargs: Additional arguments
        
    Returns:
        HotpotQAConverter instance
    """
    return HotpotQAConverter(output_dir, variant=variant, **kwargs)


def create_nq_converter(
    output_dir: Union[str, Path],
    subset: str = "validation",
    **kwargs
) -> NaturalQuestionsConverter:
    """Create a Natural Questions converter
    
    Args:
        output_dir: Output directory
        subset: Dataset subset ("validation" or "train")
        **kwargs: Additional arguments
        
    Returns:
        NaturalQuestionsConverter instance
    """
    return NaturalQuestionsConverter(output_dir, subset=subset, **kwargs)


def create_xquad_converter(
    output_dir: Union[str, Path],
    language: str = "zh",
    **kwargs
) -> XQuADConverter:
    """Create an XQuAD converter
    
    Args:
        output_dir: Output directory
        language: Language code (default: "zh" for Chinese)
        **kwargs: Additional arguments
        
    Returns:
        XQuADConverter instance
    """
    return XQuADConverter(output_dir, language=language, **kwargs)


def list_available_converters() -> Dict[str, str]:
    """List all available converters
    
    Returns:
        Dictionary mapping converter names to descriptions
    """
    return {
        "hotpotqa": "HotpotQA multi-hop question answering dataset",
        "natural_questions": "Google Natural Questions dataset",
        "xquad": "XQuAD multilingual QA dataset"
    }


__all__ = [
    "BaseConverter",
    "ConversionResult",
    "HotpotQAConverter",
    "NaturalQuestionsConverter",
    "XQuADConverter",
    "convert_dataset",
    "create_hotpotqa_converter",
    "create_nq_converter",
    "create_xquad_converter",
    "list_available_converters"
]
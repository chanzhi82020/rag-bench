"""Data schemas for Golden Dataset"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator, field_validator
import json


@dataclass
class GoldenRecord:
    """A single record in a Golden Dataset"""
    id: str = field(metadata={"description": "Unique identifier for this record"})
    user_input: str = field(metadata={"description": "User question or query"})
    reference: str = field(metadata={"description": "Reference answer"})
    reference_contexts: List[str] = field(metadata={"description": "List of relevant context passages"})
    reference_context_ids: Optional[List[str]] = field(
        default=None,
        metadata={"description": "Optional list of context IDs for reference contexts"}
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Additional metadata for this record"}
    )


class GoldenRecordModel(BaseModel):
    """Pydantic model for Golden Record validation"""
    id: str = Field(..., min_length=1, description="Unique identifier for this record")
    user_input: str = Field(..., min_length=1, description="User question or query")
    reference: str = Field(..., min_length=1, description="Reference answer")
    reference_contexts: List[str] = Field(..., min_length=1, description="List of relevant context passages")
    reference_context_ids: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of context IDs"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('id')
    def validate_id(cls, v):
        if not v.strip():
            raise ValueError('id cannot be empty')
        return v
    
    @field_validator('user_input')
    def validate_user_input(cls, v):
        if not v.strip():
            raise ValueError('user_input cannot be empty')
        if len(v) > 5000:
            raise ValueError('user_input too long (max 5000 characters)')
        return v
    
    @field_validator('reference')
    def validate_reference(cls, v):
        if not v.strip():
            raise ValueError('reference cannot be empty')
        if len(v) > 10000:
            raise ValueError('reference too long (max 10000 characters)')
        return v
    
    @field_validator('reference_contexts')
    def validate_reference_contexts(cls, v):
        if not v:
            raise ValueError('reference_contexts cannot be empty')
        for ctx in v:
            if not ctx.strip():
                raise ValueError('reference_contexts cannot contain empty strings')
        return v


@dataclass
class CorpusRecord:
    """A single document in the corpus"""
    reference_context: str = field(metadata={"description": "Document content"})
    reference_context_id: str = field(metadata={"description": "Unique document ID"})
    title: str = field(metadata={"description": "Document title"})
    metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Additional metadata for this document"}
    )


class CorpusRecordModel(BaseModel):
    """Pydantic model for Corpus Record validation"""
    reference_context: str = Field(..., min_length=1, description="Document content")
    reference_context_id: str = Field(..., min_length=1, description="Unique document ID")
    title: str = Field(..., min_length=1, description="Document title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('reference_context_id')
    def validate_context_id(cls, v):
        if not v.strip():
            raise ValueError('reference_context_id cannot be empty')
        return v


@dataclass
class DatasetMetadata:
    """Metadata for a dataset"""
    name: str
    version: str
    description: str
    source: str
    size: int
    domain: str
    language: str = "en"
    created_at: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class DatasetMetadataModel(BaseModel):
    """Pydantic model for Dataset Metadata validation"""
    name: str = Field(..., description="Dataset name")
    version: str = Field(..., description="Dataset version")
    description: str = Field(..., description="Dataset description")
    source: str = Field(..., description="Dataset source")
    size: int = Field(..., gt=0, description="Number of records")
    domain: str = Field(..., description="Domain of the dataset")
    language: str = Field(default="en", description="Language code")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    license: Optional[str] = Field(default=None, description="License information")
    tags: List[str] = Field(default_factory=list, description="Tags")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistics")


def parse_golden_record(data: Union[Dict, str]) -> GoldenRecord:
    """Parse a GoldenRecord from dict or JSON string
    
    If the 'id' field is missing from the data, a UUID will be generated.
    This provides backward compatibility during the transition period.
    """
    if isinstance(data, str):
        data = json.loads(data)
    
    # Generate ID if missing (backward compatibility)
    if 'id' not in data:
        import uuid
        data['id'] = str(uuid.uuid4())
    
    # Validate with Pydantic
    validated = GoldenRecordModel(**data)
    
    return GoldenRecord(
        id=validated.id,
        user_input=validated.user_input,
        reference=validated.reference,
        reference_contexts=validated.reference_contexts,
        reference_context_ids=validated.reference_context_ids,
        metadata=validated.metadata
    )


def parse_corpus_record(data: Union[Dict, str]) -> CorpusRecord:
    """Parse a CorpusRecord from dict or JSON string"""
    if isinstance(data, str):
        data = json.loads(data)
    
    # Validate with Pydantic
    validated = CorpusRecordModel(**data)
    
    return CorpusRecord(
        reference_context=validated.reference_context,
        reference_context_id=validated.reference_context_id,
        title=validated.title,
        metadata=validated.metadata
    )
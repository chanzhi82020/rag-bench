"""Tests for BaselineRAG persistence methods"""
import os
import tempfile
from pathlib import Path

import pytest

from rag_benchmark.prepare.baseline_rag import BaselineRAG
from rag_benchmark.prepare.rag_interface import RAGConfig


@pytest.fixture
def temp_indices_dir():
    """Provide temporary directory for index storage"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_rag_instance():
    """Provide a BaselineRAG instance for testing"""
    config = RAGConfig(top_k=3)
    return BaselineRAG(config=config)


@pytest.fixture
def sample_documents():
    """Provide sample document corpus"""
    return [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "FAISS is a library for efficient similarity search."
    ]


def test_has_index_without_indexing(sample_rag_instance):
    """Test has_index returns False when no documents are indexed"""
    assert sample_rag_instance.has_index() is False


def test_get_index_stats_without_indexing(sample_rag_instance):
    """Test get_index_stats returns correct stats when not indexed"""
    stats = sample_rag_instance.get_index_stats()
    
    assert stats["has_index"] is False
    assert stats["document_count"] == 0
    assert stats["embedding_dimension"] is None
    assert stats["index_type"] is None


def test_load_from_disk_nonexistent_path(sample_rag_instance, temp_indices_dir):
    """Test load_from_disk returns False for nonexistent path"""
    nonexistent_path = temp_indices_dir / "nonexistent_rag"
    success = sample_rag_instance.load_from_disk(nonexistent_path)
    
    assert success is False
    assert sample_rag_instance.has_index() is False


@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not installed"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_has_index_after_indexing(sample_documents):
    """Test has_index returns True after indexing documents"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=3)
        )
        
        rag.index_documents(sample_documents)
        assert rag.has_index() is True
        
    except ImportError:
        pytest.skip("langchain-openai not installed")


@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not installed"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_get_index_stats_after_indexing(sample_documents):
    """Test get_index_stats returns correct stats after indexing"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=3)
        )
        
        rag.index_documents(sample_documents)
        stats = rag.get_index_stats()
        
        assert stats["has_index"] is True
        assert stats["document_count"] == len(sample_documents)
        assert stats["embedding_dimension"] is not None
        assert stats["embedding_dimension"] > 0
        assert stats["index_type"] is not None
        
    except ImportError:
        pytest.skip("langchain-openai not installed")


@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not installed"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_save_to_disk(sample_documents, temp_indices_dir):
    """Test save_to_disk creates all required files"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=3)
        )
        
        rag.index_documents(sample_documents)
        
        save_path = temp_indices_dir / "test_rag"
        rag.save_to_disk(save_path)
        
        # Verify all files exist
        assert (save_path / "index.faiss").exists()
        assert (save_path / "corpus.json").exists()
        assert (save_path / "embeddings.npy").exists()
        assert (save_path / "metadata.json").exists()
        
    except ImportError:
        pytest.skip("langchain-openai not installed")


@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not installed"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_load_from_disk(sample_documents, temp_indices_dir):
    """Test load_from_disk restores index data"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        config = RAGConfig(top_k=3)
        
        rag = BaselineRAG(
            embedding_model=embedding_model,
            llm=llm,
            config=config
        )
        
        # Index and save
        rag.index_documents(sample_documents)
        save_path = temp_indices_dir / "test_rag"
        rag.save_to_disk(save_path)
        
        # Create new instance and load
        new_rag = BaselineRAG(
            embedding_model=embedding_model,
            llm=llm,
            config=config
        )
        
        success = new_rag.load_from_disk(save_path)
        
        assert success is True
        assert new_rag.has_index() is True
        assert len(new_rag.documents) == len(sample_documents)
        assert new_rag.documents == sample_documents
        
    except ImportError:
        pytest.skip("langchain-openai not installed")


@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not installed"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_save_load_round_trip(sample_documents, temp_indices_dir):
    """Test that save and load preserve index functionality"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        config = RAGConfig(top_k=3)
        
        rag = BaselineRAG(
            embedding_model=embedding_model,
            llm=llm,
            config=config
        )
        
        # Index documents
        rag.index_documents(sample_documents)
        
        # Get original query result
        original_result = rag.retrieve("Python programming", top_k=2)
        
        # Save to disk
        save_path = temp_indices_dir / "test_rag"
        rag.save_to_disk(save_path)
        
        # Create new instance and load
        new_rag = BaselineRAG(
            embedding_model=embedding_model,
            llm=llm,
            config=config
        )
        new_rag.load_from_disk(save_path)
        
        # Get new query result
        new_result = new_rag.retrieve("Python programming", top_k=2)
        
        # Verify results are similar (same contexts)
        assert new_result.contexts == original_result.contexts
        assert len(new_result.scores) == len(original_result.scores)
        
    except ImportError:
        pytest.skip("langchain-openai not installed")

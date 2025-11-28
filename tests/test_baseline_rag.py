"""测试Baseline RAG"""
import os

import pytest

from rag_benchmark.prepare import BaselineRAG, RAGConfig


def test_baseline_rag_init():
    """测试BaselineRAG初始化"""
    rag = BaselineRAG()
    
    assert rag.config is not None
    assert rag.index is None
    assert rag.documents == []


def test_baseline_rag_with_config():
    """测试使用自定义配置初始化"""
    config = RAGConfig(top_k=5)
    rag = BaselineRAG(config=config)
    
    assert rag.config.top_k == 5


@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not installed"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_baseline_rag_index_documents():
    """测试文档索引功能"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=3)
        )
        
        # 索引测试文档
        documents = [
            "Python是一种高级编程语言",
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络"
        ]
        
        rag.index_documents(documents)
        
        # 验证索引
        assert rag.index is not None
        assert len(rag.documents) == 3
        assert rag.index.ntotal == 3
        
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
@pytest.mark.asyncio
async def test_baseline_rag_retrieve():
    """测试检索功能"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=2)
        )
        
        # 索引文档
        documents = [
            "Python是一种高级编程语言",
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络"
        ]
        rag.index_documents(documents)
        
        # 检索
        result = await rag.retrieve("什么是Python", top_k=2)
        
        # 验证结果
        assert len(result.contexts) == 2
        assert all(isinstance(doc, str) for doc in result.contexts)
        assert result.scores is not None
        assert len(result.scores) == 2
        
    except ImportError:
        pytest.skip("langchain-openai not installed")


@pytest.mark.asyncio
async def test_baseline_rag_retrieve_without_index():
    """测试未索引时的检索"""
    rag = BaselineRAG()
    
    # 未索引时应返回空结果
    result = await rag.retrieve("test query")
    assert result.contexts == []


@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not installed"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
@pytest.mark.asyncio
async def test_baseline_rag_generate():
    """测试生成功能"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=2)
        )
        
        # 测试生成
        query = "什么是Python？"
        contexts = ["Python是一种高级编程语言", "Python广泛用于数据科学"]
        
        result = await rag.generate(query, contexts)
        
        # 验证答案
        assert isinstance(result.response, str)
        assert len(result.response) > 0
        
    except ImportError:
        pytest.skip("langchain-openai not installed")

@pytest.mark.asyncio
async def test_baseline_rag_generate_empty_contexts():
    """测试空上下文时的生成"""
    rag = BaselineRAG()
    
    result = await rag.generate("test query", [])
    
    # 应返回默认消息
    assert "没有找到相关信息" in result.response or "不知道" in result.response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



@pytest.mark.skipif(
    not pytest.importorskip("faiss", reason="FAISS not installed"),
    reason="FAISS not installed"
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
@pytest.mark.asyncio
async def test_baseline_rag_batch_retrieve():
    """测试批量检索功能"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=2)
        )
        
        # 索引文档
        documents = [
            "Python是一种高级编程语言",
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络"
        ]
        rag.index_documents(documents)
        
        # 批量检索
        queries = ["什么是Python", "什么是机器学习"]
        results =await rag.batch_retrieve(queries, top_k=2)
        
        # 验证结果
        assert len(results) == 2
        for result in results:
            assert len(result.contexts) == 2
            assert result.scores is not None
            assert len(result.scores) == 2
        
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
@pytest.mark.asyncio
async def test_baseline_rag_batch_generate():
    """测试批量生成功能"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=2)
        )
        
        # 测试批量生成
        queries = ["什么是Python？", "什么是机器学习？"]
        contexts_list = [
            ["Python是一种高级编程语言", "Python广泛用于数据科学"],
            ["机器学习是人工智能的一个分支", "机器学习使用算法学习数据"]
        ]
        
        results = await rag.batch_generate(queries, contexts_list)
        
        # 验证结果
        assert len(results) == 2
        for result in results:
            assert isinstance(result.response, str)
            assert len(result.response) > 0
        
    except ImportError:
        pytest.skip("langchain-openai not installed")

@pytest.mark.asyncio
async def test_baseline_rag_batch_retrieve_without_index():
    """测试未索引时的批量检索"""
    rag = BaselineRAG()
    
    # 未索引时应返回空结果列表
    queries = ["query1", "query2"]
    results = await rag.batch_retrieve(queries)
    
    assert len(results) == 2
    for result in results:
        assert result.contexts == []


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
@pytest.mark.asyncio
async def test_baseline_rag_batch_generate_empty_contexts():
    """测试空上下文时的批量生成"""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        rag = BaselineRAG(
            embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            config=RAGConfig(top_k=2)
        )
        
        # 测试空上下文
        queries = ["query1", "query2"]
        contexts_list = [[], []]
        
        results = await rag.batch_generate(queries, contexts_list)
        
        # 验证结果
        assert len(results) == 2
        for result in results:
            assert "没有找到相关信息" in result.response or "不知道" in result.response.lower()
        
    except ImportError:
        pytest.skip("langchain-openai not installed")

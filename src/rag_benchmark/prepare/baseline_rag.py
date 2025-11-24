"""Baseline RAG实现

使用FAISS作为向量检索器，支持自定义LLM的简单RAG系统。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

from .rag_interface import RAGConfig, RAGInterface, RetrievalResult, GenerationResult

logger = logging.getLogger(__name__)


class BaselineRAG(RAGInterface):
    """Baseline RAG实现
    
    使用FAISS进行向量检索，支持自定义LLM进行生成。
    
    Features:
    - FAISS向量检索
    - 支持自定义Embedding模型
    - 支持自定义LLM
    - 简单的prompt模板
    
    """

    def __init__(
            self,
            embedding_model: Optional[OpenAIEmbeddings] = None,
            llm: Optional[BaseChatModel] = None,
            config: Optional[RAGConfig] = None
    ):
        """初始化Baseline RAG
        
        Args:
            embedding_model: Langchain Embedding模型实例
            llm: Langchain LLM模型实例
            config: RAG配置，如果为None则使用默认配置
        """
        super().__init__(config)
        self.embedding_model = embedding_model
        self.llm = llm
        self.index = None
        self.documents = []
        self.embeddings_cache = None

        # 延迟导入以避免强制依赖
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            logger.warning(
                "FAISS未安装，无法使用向量检索功能。"
                "请安装: pip install faiss-cpu 或 pip install faiss-gpu"
            )
            self.faiss = None

        # 不自动创建默认模型，让用户显式提供
        if self.embedding_model is None:
            logger.warning(
                "Embedding模型未设置。请在创建BaselineRAG时传入embedding_model参数。"
            )
        if self.llm is None:
            logger.warning(
                "LLM模型未设置。请在创建BaselineRAG时传入llm参数。"
            )

    def index_documents(self, documents: List[str], batch_size: int = 50):
        """索引文档到FAISS（支持批量处理）
        
        Args:
            documents: 文档列表
            batch_size: 批量处理大小，默认50（避免超过API限制）
        """
        if self.faiss is None:
            raise RuntimeError("FAISS未安装，无法索引文档")

        if self.embedding_model is None:
            raise RuntimeError("Embedding模型未设置")

        logger.info(f"开始索引 {len(documents)} 个文档（批量大小: {batch_size}）...")

        self.documents = documents

        # 批量生成embeddings以避免API限制
        all_embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"处理批次 {batch_num}/{total_batches}，文档 {i+1}-{min(i+batch_size, len(documents))}")
            
            try:
                batch_embeddings = self.embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"批次 {batch_num} 生成embeddings失败: {e}")
                raise
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        self.embeddings_cache = embeddings_array

        # 创建FAISS索引
        dimension = embeddings_array.shape[1]
        self.index = self.faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

        logger.info(f"索引完成，共 {self.index.ntotal} 个文档")

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回top-k个结果，默认使用config中的值
            
        Returns:
            RetrievalResult实例
        """
        if self.index is None:
            logger.warning("索引未初始化，返回空结果")
            return RetrievalResult(contexts=[])

        if self.embedding_model is None:
            raise RuntimeError("Embedding模型未设置")

        if top_k is None:
            top_k = self.config.top_k

        # 生成query embedding
        query_embedding = await self.embedding_model.aembed_query(query)
        query_array = np.array([query_embedding], dtype=np.float32)

        # 检索
        distances, indices = self.index.search(query_array, top_k)

        # 返回文档
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        scores = [float(1.0 / (1.0 + dist)) for dist in distances[0]]  # 转换距离为相似度分数

        logger.debug(f"检索到 {len(retrieved_docs)} 个文档")
        return RetrievalResult(
            contexts=retrieved_docs,
            scores=scores
        )

    async def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        """基于上下文生成答案
        
        Args:
            query: 用户问题
            contexts: 检索到的上下文列表
            
        Returns:
            GenerationResult实例
        """
        if not contexts:
            return GenerationResult(
                response="抱歉，我没有找到相关信息来回答这个问题。"
            )

        if self.llm is None:
            raise RuntimeError("LLM模型未设置")

        # 构建prompt
        context_str = "\n\n".join([f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文信息：
{context_str}

问题：{query}

答案："""

        # 调用LLM生成
        try:
            response = await self.llm.ainvoke(prompt)
            # 处理不同类型的响应
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)

            logger.debug(f"生成答案: {answer[:100]}...")
            return GenerationResult(response=answer)
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return GenerationResult(
                response=f"生成答案时出错: {str(e)}"
            )

    async def batch_retrieve(
            self, queries: List[str], top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """批量检索（优化版本）
        
        使用批量embedding生成和FAISS批量搜索来提高性能。
        
        Args:
            queries: 查询列表
            top_k: 返回top-k个结果
            
        Returns:
            每个查询对应的RetrievalResult列表
            
        """
        if self.index is None:
            logger.warning("索引未初始化，返回空结果")
            return [RetrievalResult(contexts=[]) for _ in queries]

        if self.embedding_model is None:
            raise RuntimeError("Embedding模型未设置")

        if top_k is None:
            top_k = self.config.top_k

        # 批量生成query embeddings（一次性生成所有查询的embeddings）
        logger.debug(f"批量生成 {len(queries)} 个查询的embeddings...")
        query_embeddings = await self.embedding_model.aembed_documents(queries)
        query_array = np.array(query_embeddings, dtype=np.float32)

        # 批量检索（FAISS支持批量搜索）
        distances, indices = self.index.search(query_array, top_k)

        # 构建结果
        results = []
        for i in range(len(queries)):
            retrieved_docs = [self.documents[idx] for idx in indices[i]]
            scores = [float(1.0 / (1.0 + dist)) for dist in distances[i]]
            results.append(RetrievalResult(contexts=retrieved_docs, scores=scores))

        logger.debug(f"批量检索完成，共 {len(results)} 个结果")
        return results

    async def batch_generate(
            self, queries: List[str], contexts_list: List[List[str]]
    ) -> List[GenerationResult]:
        """批量生成（优化版本）
        
        使用LLM的batch方法进行批量生成。
        
        Args:
            queries: 查询列表
            contexts_list: 每个查询对应的上下文列表
            
        Returns:
            每个查询对应的GenerationResult列表
            
        """
        if self.llm is None:
            raise RuntimeError("LLM模型未设置")

        # 构建批量prompts
        prompts = []
        for query, contexts in zip(queries, contexts_list):
            if not contexts:
                # 空上下文，使用默认prompt
                prompt = f"""问题：{query}

答案：抱歉，我没有找到相关信息来回答这个问题。"""
            else:
                context_str = "\n\n".join([f"[{j + 1}] {ctx}" for j, ctx in enumerate(contexts)])
                prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文信息：
{context_str}

问题：{query}

答案："""
            prompts.append(prompt)

        # 批量调用LLM
        logger.debug(f"批量生成 {len(prompts)} 个答案...")

        try:
            # 使用batch方法批量调用
            responses = await self.llm.abatch(prompts)

            # 处理响应
            results = []
            for response in responses:
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
                results.append(GenerationResult(response=answer))

            logger.debug(f"批量生成完成，共 {len(results)} 个答案")
            return results

        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            # 返回错误结果
            return [GenerationResult(
                response=f"生成答案时出错: {str(e)}"
            ) for _ in queries]


    def save_to_disk(self, save_path: Path) -> None:
        """Save index data to specified path
        
        This method persists the FAISS index, document corpus, embeddings cache,
        and metadata to disk for later restoration.
        
        Args:
            save_path: Directory path where index data should be saved
            
        Raises:
            RuntimeError: If persistence fails or RAG has no indexed data
            
        """
        from ..persistence import save_index_data
        
        # Extract RAG name from path (last component)
        rag_name = save_path.name
        indices_dir = save_path.parent
        
        save_index_data(rag_name, self, indices_dir)
        logger.info(f"Index data saved to {save_path}")

    def load_from_disk(self, load_path: Path) -> bool:
        """Load index data from specified path
        
        This method restores the FAISS index, document corpus, and embeddings cache
        from previously persisted data.
        
        Args:
            load_path: Directory path where index data is stored
            
        Returns:
            True if loading succeeded, False otherwise
            
        """
        from ..persistence import load_index_data
        
        # Extract RAG name from path (last component)
        rag_name = load_path.name
        indices_dir = load_path.parent
        
        success = load_index_data(rag_name, self, indices_dir)
        if success:
            logger.info(f"Index data loaded from {load_path}")
        else:
            logger.warning(f"Failed to load index data from {load_path}")
        
        return success

    def delete_from_disk(self, delete_path: Path) -> None:
        """Delete persisted index data from disk
        
        This method removes all persisted files (FAISS index, document corpus,
        embeddings cache, and metadata) for this RAG instance.
        
        Args:
            delete_path: Directory path where index data is stored
            
        """
        from ..persistence import delete_index_data
        
        # Extract RAG name from path (last component)
        rag_name = delete_path.name
        indices_dir = delete_path.parent
        
        delete_index_data(rag_name, indices_dir)
        logger.info(f"Index data deleted from {delete_path}")

    def has_index(self) -> bool:
        """Check if RAG instance has indexed documents
        
        Returns:
            True if the instance has a FAISS index and documents, False otherwise
            
        """
        return self.index is not None and len(self.documents) > 0

    def get_index_stats(self) -> Dict:
        """Get statistics about current index
        
        Returns:
            Dictionary containing index statistics including:
            - has_index: Whether the instance has indexed documents
            - document_count: Number of indexed documents (if indexed)
            - embedding_dimension: Dimension of embeddings (if indexed)
            - index_type: Type of FAISS index (if indexed)
            
        """
        if not self.has_index():
            return {
                "has_index": False,
                "document_count": 0,
                "embedding_dimension": None,
                "index_type": None
            }
        
        # Get embedding dimension from embeddings cache or index
        embedding_dim = None
        if self.embeddings_cache is not None:
            embedding_dim = self.embeddings_cache.shape[1]
        elif self.index is not None:
            embedding_dim = self.index.d
        
        # Get index type
        index_type = type(self.index).__name__ if self.index is not None else None
        
        return {
            "has_index": True,
            "document_count": len(self.documents),
            "embedding_dimension": embedding_dim,
            "index_type": index_type
        }

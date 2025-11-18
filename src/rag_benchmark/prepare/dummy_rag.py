"""示例RAG实现，用于测试和演示"""

import random
from typing import List, Optional

from .rag_interface import RAGConfig, RAGInterface, RetrievalResult, GenerationResult


class DummyRAG(RAGInterface):
    """虚拟RAG系统，返回模拟数据

    用于测试和演示，不依赖任何外部服务。
    检索返回固定的模拟上下文，生成返回简单的模板答案。

    Example:
        >>> rag = DummyRAG()
        >>> contexts = rag.retrieve("What is Python?")
        >>> answer = rag.generate("What is Python?", contexts)
    """

    def __init__(self, config: Optional[RAGConfig] = None, seed: Optional[int] = None):
        """初始化DummyRAG

        Args:
            config: RAG配置
            seed: 随机种子，用于可复现的结果
        """
        super().__init__(config)
        if seed is not None:
            random.seed(seed)

        # 预定义的模拟上下文
        self.mock_contexts = [
            "This is a mock context about the topic.",
            "Here is some relevant information from the knowledge base.",
            "Additional context that might be helpful for answering.",
            "Background information related to the query.",
            "Supporting evidence from the corpus.",
        ]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """检索模拟上下文

        Args:
            query: 用户查询
            top_k: 返回top-k个结果

        Returns:
            RetrievalResult包含模拟的上下文和元数据
        """
        k = top_k if top_k is not None else self.config.top_k
        k = min(k, len(self.mock_contexts))

        # 返回前k个模拟上下文
        contexts = self.mock_contexts[:k]
        
        # 生成模拟的context_ids和scores
        context_ids = [f"dummy_ctx_{i}" for i in range(len(contexts))]
        scores = [0.95 - i * 0.05 for i in range(len(contexts))]  # 递减的分数
        
        return RetrievalResult(
            contexts=contexts,
            context_ids=context_ids,
            scores=scores,
            metadata={"retrieval_method": "dummy", "query_length": len(query)}
        )

    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        """生成模拟答案

        Args:
            query: 用户查询
            contexts: 检索到的上下文

        Returns:
            GenerationResult包含模拟的答案和元数据
        """
        # 简单的模板答案
        answer = f"Based on the provided contexts, here is the answer to '{query}': "
        answer += "This is a dummy response generated for testing purposes. "
        answer += (
            f"The system retrieved {len(contexts)} contexts to formulate this answer."
        )
        
        # 生成多个候选答案
        multi_responses = [
            answer,
            f"Alternative answer 1: {answer[:50]}...",
            f"Alternative answer 2: {answer[:30]}...",
        ]
        
        return GenerationResult(
            response=answer,
            multi_responses=multi_responses,
            confidence=0.85,
            metadata={"generation_method": "dummy", "context_count": len(contexts)}
        )


class SimpleRAG(RAGInterface):
    """简单的RAG实现，基于BM25检索和模板生成

    这是一个更接近真实RAG的简单实现，但仍然不依赖外部模型。
    使用简单的关键词匹配进行检索，使用模板进行生成。

    Example:
        >>> corpus = ["Python is a programming language.", "Java is also popular."]
        >>> rag = SimpleRAG(corpus)
        >>> contexts = rag.retrieve("What is Python?")
        >>> answer = rag.generate("What is Python?", contexts)
    """

    def __init__(self, corpus: List[str], config: Optional[RAGConfig] = None):
        """初始化SimpleRAG

        Args:
            corpus: 文档语料库
            config: RAG配置
        """
        super().__init__(config)
        self.corpus = corpus

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """基于关键词匹配检索上下文

        Args:
            query: 用户查询
            top_k: 返回top-k个结果

        Returns:
            RetrievalResult包含检索到的上下文和元数据
        """
        k = top_k if top_k is not None else self.config.top_k

        # 简单的关键词匹配评分
        query_terms = set(query.lower().split())

        scored_docs = []
        for idx, doc in enumerate(self.corpus):
            doc_terms = set(doc.lower().split())
            # 计算交集大小作为相关性分数
            score = len(query_terms & doc_terms)
            scored_docs.append((score, doc, idx))

        # 按分数排序并返回top-k
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # 过滤掉分数为0的文档
        relevant_docs = [(score, doc, idx) for score, doc, idx in scored_docs if score > 0]

        # 如果没有相关文档，返回前k个文档
        if not relevant_docs:
            relevant_docs = [(0, doc, idx) for idx, doc in enumerate(self.corpus[:k])]

        # 提取top-k
        top_results = relevant_docs[:k]
        contexts = [doc for _, doc, _ in top_results]
        context_ids = [f"corpus_doc_{idx}" for _, _, idx in top_results]
        scores = [float(score) / max(len(query_terms), 1) for score, _, _ in top_results]  # 归一化分数
        
        return RetrievalResult(
            contexts=contexts,
            context_ids=context_ids,
            scores=scores,
            metadata={"retrieval_method": "keyword_matching", "corpus_size": len(self.corpus)}
        )

    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        """基于模板生成答案

        Args:
            query: 用户查询
            contexts: 检索到的上下文

        Returns:
            GenerationResult包含生成的答案和元数据
        """
        if not contexts:
            return GenerationResult(
                response="I don't have enough information to answer this question.",
                confidence=0.0,
                metadata={"generation_method": "template", "context_count": 0}
            )

        # 简单的模板生成
        answer = "Based on the available information: "

        # 使用第一个上下文作为主要信息源
        main_context = contexts[0]

        # 提取关键句子（简单实现：取第一句）
        sentences = main_context.split(".")
        if sentences:
            key_info = sentences[0].strip()
            answer += key_info
            if not key_info.endswith("."):
                answer += "."

        # 如果有多个上下文，添加补充信息
        if len(contexts) > 1:
            answer += " Additionally, "
            second_context = contexts[1]
            second_sentences = second_context.split(".")
            if second_sentences:
                answer += second_sentences[0].strip()
                if not answer.endswith("."):
                    answer += "."
        
        # 计算简单的置信度（基于上下文数量）
        confidence = min(0.5 + len(contexts) * 0.1, 0.95)
        
        return GenerationResult(
            response=answer,
            confidence=confidence,
            metadata={"generation_method": "template", "context_count": len(contexts)}
        )

    def batch_retrieve(
        self, queries: List[str], top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """批量检索（优化实现）

        Args:
            queries: 查询列表
            top_k: 返回top-k个结果

        Returns:
            每个查询对应的RetrievalResult列表
        """
        # 对于SimpleRAG，批量处理和单独处理性能相同
        # 但这里展示了如何重写批量方法
        return [self.retrieve(query, top_k) for query in queries]

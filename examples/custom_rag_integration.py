"""示例：集成自定义RAG系统

演示如何实现RAGInterface接口来集成用户自己的RAG系统
"""

from typing import List, Optional
from rag_benchmark.prepare import (
    RAGInterface,
    RAGConfig,
    RetrievalResult,
    GenerationResult,
)


class MyCustomRAG(RAGInterface):
    """自定义RAG系统示例
    
    这个示例展示了如何实现RAGInterface接口来集成你自己的RAG系统。
    你需要实现两个核心方法：retrieve() 和 generate()
    """
    
    def __init__(
        self,
        retriever_endpoint: str,
        generator_endpoint: str,
        config: Optional[RAGConfig] = None
    ):
        """初始化自定义RAG系统
        
        Args:
            retriever_endpoint: 检索服务的API端点
            generator_endpoint: 生成服务的API端点
            config: RAG配置
        """
        super().__init__(config)
        self.retriever_endpoint = retriever_endpoint
        self.generator_endpoint = generator_endpoint
        
        # 这里可以初始化你的客户端、模型等
        # 例如：
        # self.retriever_client = RetrieverClient(retriever_endpoint)
        # self.generator_client = GeneratorClient(generator_endpoint)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """实现检索逻辑
        
        Args:
            query: 用户查询
            top_k: 返回top-k个结果
            
        Returns:
            RetrievalResult包含上下文和元数据
        """
        k = top_k if top_k is not None else self.config.top_k
        
        # 实现你的检索逻辑
        # 例如：
        # response = self.retriever_client.search(
        #     query=query,
        #     top_k=k,
        #     threshold=self.config.similarity_threshold
        # )
        # contexts = [doc.text for doc in response.documents]
        # context_ids = [doc.id for doc in response.documents]
        # scores = [doc.score for doc in response.documents]
        # return RetrievalResult(
        #     contexts=contexts,
        #     context_ids=context_ids,
        #     scores=scores,
        #     metadata={"retrieval_time": response.time}
        # )
        
        # 这里是模拟实现
        print(f"Retrieving top-{k} contexts for query: {query[:50]}...")
        contexts = [
            f"Retrieved context {i+1} for query: {query[:30]}..."
            for i in range(k)
        ]
        context_ids = [f"doc_{i}" for i in range(k)]
        scores = [0.95 - i * 0.05 for i in range(k)]
        
        return RetrievalResult(
            contexts=contexts,
            context_ids=context_ids,
            scores=scores,
            metadata={"retrieval_method": "custom"}
        )
    
    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        """实现生成逻辑
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文列表
            
        Returns:
            GenerationResult包含答案和元数据
        """
        # 实现你的生成逻辑
        # 例如：
        # prompt = self._build_prompt(query, contexts)
        # response = self.generator_client.generate(
        #     prompt=prompt,
        #     max_length=self.config.max_length,
        #     temperature=self.config.temperature,
        #     top_p=self.config.top_p
        # )
        # return GenerationResult(
        #     response=response.text,
        #     confidence=response.confidence,
        #     metadata={"generation_time": response.time}
        # )
        
        # 这里是模拟实现
        print(f"Generating answer for query: {query[:50]}...")
        answer = f"Generated answer based on {len(contexts)} contexts: ..."
        
        return GenerationResult(
            response=answer,
            confidence=0.88,
            metadata={"generation_method": "custom"}
        )
    
    def batch_retrieve(self, queries: List[str], top_k: Optional[int] = None) -> List[RetrievalResult]:
        """批量检索（可选优化）
        
        如果你的检索服务支持批量请求，可以重写这个方法来提高性能。
        """
        # 如果支持批量API
        # response = self.retriever_client.batch_search(queries, top_k)
        # return [
        #     RetrievalResult(
        #         contexts=docs.texts,
        #         context_ids=docs.ids,
        #         scores=docs.scores
        #     )
        #     for docs in response.documents_list
        # ]
        
        # 否则使用默认实现（逐条处理）
        return super().batch_retrieve(queries, top_k)
    
    def batch_generate(self, queries: List[str], contexts_list: List[List[str]]) -> List[GenerationResult]:
        """批量生成（可选优化）
        
        如果你的生成服务支持批量请求，可以重写这个方法来提高性能。
        """
        # 如果支持批量API
        # prompts = [self._build_prompt(q, c) for q, c in zip(queries, contexts_list)]
        # response = self.generator_client.batch_generate(prompts)
        # return [
        #     GenerationResult(response=text, confidence=conf)
        #     for text, conf in zip(response.texts, response.confidences)
        # ]
        
        # 否则使用默认实现（逐条处理）
        return super().batch_generate(queries, contexts_list)


class VectorDBRAG(RAGInterface):
    """基于向量数据库的RAG系统示例
    
    展示如何集成FAISS、Chroma等向量数据库
    """
    
    def __init__(
        self,
        vector_db,  # 你的向量数据库实例
        embedding_model,  # 你的embedding模型
        llm_model,  # 你的LLM模型
        config: Optional[RAGConfig] = None
    ):
        """初始化基于向量数据库的RAG
        
        Args:
            vector_db: 向量数据库实例（如FAISS index, Chroma collection等）
            embedding_model: Embedding模型
            llm_model: LLM模型
            config: RAG配置
        """
        super().__init__(config)
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.llm_model = llm_model
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """使用向量数据库检索
        
        Args:
            query: 用户查询
            top_k: 返回top-k个结果
            
        Returns:
            RetrievalResult包含上下文和元数据
        """
        k = top_k if top_k is not None else self.config.top_k
        
        # 1. 将查询转换为向量
        # query_embedding = self.embedding_model.encode(query)
        
        # 2. 在向量数据库中搜索
        # results = self.vector_db.search(
        #     query_embedding,
        #     k=k,
        #     threshold=self.config.similarity_threshold
        # )
        
        # 3. 返回结果
        # return RetrievalResult(
        #     contexts=[doc.text for doc in results],
        #     context_ids=[doc.id for doc in results],
        #     scores=[doc.score for doc in results],
        #     metadata={"vector_db": "faiss", "embedding_model": "sentence-transformers"}
        # )
        
        # 模拟实现
        contexts = [f"Vector DB context {i+1}" for i in range(k)]
        context_ids = [f"vec_doc_{i}" for i in range(k)]
        scores = [0.92 - i * 0.03 for i in range(k)]
        
        return RetrievalResult(
            contexts=contexts,
            context_ids=context_ids,
            scores=scores,
            metadata={"vector_db": "simulated"}
        )
    
    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        """使用LLM生成答案
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文列表
            
        Returns:
            GenerationResult包含答案和元数据
        """
        # 1. 构建prompt
        # prompt = f"Question: {query}\n\nContexts:\n"
        # for i, ctx in enumerate(contexts, 1):
        #     prompt += f"{i}. {ctx}\n"
        # prompt += "\nAnswer:"
        
        # 2. 调用LLM生成
        # response = self.llm_model.generate(
        #     prompt,
        #     max_length=self.config.max_length,
        #     temperature=self.config.temperature,
        #     top_p=self.config.top_p
        # )
        
        # 3. 返回结果
        # return GenerationResult(
        #     response=response.text,
        #     confidence=response.confidence,
        #     metadata={"model": "llama-2", "tokens": response.token_count}
        # )
        
        # 模拟实现
        answer = f"LLM generated answer based on {len(contexts)} contexts"
        
        return GenerationResult(
            response=answer,
            confidence=0.91,
            metadata={"model": "simulated"}
        )


class APIBasedRAG(RAGInterface):
    """基于API的RAG系统示例
    
    展示如何集成OpenAI、Anthropic等API服务
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        model_name: str = "gpt-3.5-turbo",
        config: Optional[RAGConfig] = None
    ):
        """初始化基于API的RAG
        
        Args:
            api_key: API密钥
            index_name: 索引名称
            model_name: 模型名称
            config: RAG配置
        """
        super().__init__(config)
        self.api_key = api_key
        self.index_name = index_name
        self.model_name = model_name
        
        # 初始化API客户端
        # self.client = OpenAI(api_key=api_key)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """使用API检索
        
        可以使用向量数据库API（如Pinecone, Weaviate）
        或搜索API（如Elasticsearch）
        """
        k = top_k if top_k is not None else self.config.top_k
        
        # 调用检索API
        # response = self.client.search(
        #     index=self.index_name,
        #     query=query,
        #     top_k=k
        # )
        # return RetrievalResult(
        #     contexts=[hit.text for hit in response.hits],
        #     context_ids=[hit.id for hit in response.hits],
        #     scores=[hit.score for hit in response.hits],
        #     metadata={"index": self.index_name, "api": "pinecone"}
        # )
        
        # 模拟实现
        contexts = [f"API retrieved context {i+1}" for i in range(k)]
        context_ids = [f"api_doc_{i}" for i in range(k)]
        scores = [0.90 - i * 0.04 for i in range(k)]
        
        return RetrievalResult(
            contexts=contexts,
            context_ids=context_ids,
            scores=scores,
            metadata={"api": "simulated"}
        )
    
    def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        """使用API生成答案"""
        # 构建消息
        # context_text = "\n\n".join(contexts)
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        # ]
        
        # 调用生成API
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     max_tokens=self.config.max_length,
        #     temperature=self.config.temperature,
        #     top_p=self.config.top_p
        # )
        # return GenerationResult(
        #     response=response.choices[0].message.content,
        #     confidence=response.confidence if hasattr(response, 'confidence') else None,
        #     metadata={
        #         "model": self.model_name,
        #         "tokens": response.usage.total_tokens
        #     }
        # )
        
        # 模拟实现
        answer = f"API generated answer using {self.model_name}"
        
        return GenerationResult(
            response=answer,
            confidence=0.93,
            metadata={"model": self.model_name}
        )


def example_usage():
    """演示如何使用自定义RAG"""
    print("=" * 80)
    print("Custom RAG Integration Examples")
    print("=" * 80)
    
    # 示例1：使用自定义RAG
    print("\n1. MyCustomRAG Example:")
    config = RAGConfig(top_k=3, max_length=256, temperature=0.7)
    rag = MyCustomRAG(
        retriever_endpoint="http://localhost:8000/retrieve",
        generator_endpoint="http://localhost:8001/generate",
        config=config
    )
    
    query = "What is machine learning?"
    retrieval_result = rag.retrieve(query)
    generation_result = rag.generate(query, retrieval_result.contexts)
    print(f"   Query: {query}")
    print(f"   Contexts: {retrieval_result.contexts}")
    print(f"   Context IDs: {retrieval_result.context_ids}")
    print(f"   Scores: {retrieval_result.scores}")
    print(f"   Answer: {generation_result.response}")
    print(f"   Confidence: {generation_result.confidence}")
    
    # 示例2：使用便捷方法
    print("\n2. Using retrieve_and_generate:")
    retrieval_result, generation_result = rag.retrieve_and_generate(query)
    print(f"   Retrieved {len(retrieval_result.contexts)} contexts")
    print(f"   Generated answer: {generation_result.response}")
    
    # 示例3：批量处理
    print("\n3. Batch processing:")
    queries = [
        "What is Python?",
        "What is deep learning?",
        "What is NLP?"
    ]
    results = rag.batch_retrieve_and_generate(queries)
    for q, (retrieval_res, generation_res) in zip(queries, results):
        print(f"   Q: {q}")
        print(f"   A: {generation_res.response}")
    
    print("\n" + "=" * 80)
    print("Integration examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()

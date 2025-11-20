"""批量处理性能对比示例

演示BaselineRAG的批量处理功能如何提升性能。
"""

import time
from pathlib import Path

from pydantic import SecretStr

from rag_benchmark.datasets import GoldenDataset
from rag_benchmark.datasets.loaders.base import BaseLoader
from rag_benchmark.prepare import BaselineRAG, RAGConfig

# 创建输出目录
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("批量处理性能对比示例")
    print("=" * 60)
    
    print("\n注意：此示例需要以下依赖：")
    print("  - faiss-cpu 或 faiss-gpu")
    print("  - langchain-openai")
    print("  - OPENAI_API_KEY环境变量")
    print("=" * 60)
    
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = ChatOpenAI(
            model="deepseek-ai/deepseek-v3.1",
            api_key=SecretStr("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"),
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.0,
        )
        # Create 智谱 Embedding model using langchain
        embedding = OpenAIEmbeddings(
            model="embedding-3",
            api_key=SecretStr("7f08f66caad549708238a57e0f7f33f7.EfQ9HoYpYZqBCRFX"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )

        # 1. 加载数据集
        print("\n1. 加载数据集...")
        golden_ds = GoldenDataset("xquad", subset="zh")
        records = golden_ds.head(20)  # 使用20条记录进行测试
        print(f"   加载了 {len(records)} 条记录")
        
        # 2. 创建BaselineRAG
        print("\n2. 创建BaselineRAG...")
        rag = BaselineRAG(
            embedding_model=embedding,
            llm=llm,
            config=RAGConfig(top_k=3)
        )
        print("   ✓ BaselineRAG创建成功")
        
        # 3. 索引文档
        print("\n3. 索引文档...")
        corpus = []
        for record in records:
            if record.reference_contexts:
                corpus.extend(record.reference_contexts)
        corpus = list(set(corpus))
        
        rag.index_documents(corpus)
        print(f"   ✓ 索引完成，共 {len(corpus)} 个文档")
        
        # 4. 准备查询
        queries = [record.user_input for record in records[:10]]  # 使用前10个查询
        print(f"\n4. 准备 {len(queries)} 个查询进行测试")
        
        # 5. 测试逐个检索
        print("\n5. 测试逐个检索...")
        start_time = time.time()
        individual_results = []
        for query in queries:
            result = rag.retrieve(query, top_k=3)
            individual_results.append(result)
        individual_time = time.time() - start_time
        print(f"   逐个检索耗时: {individual_time:.2f}秒")
        print(f"   平均每个查询: {individual_time/len(queries):.3f}秒")
        
        # 6. 测试批量检索
        print("\n6. 测试批量检索...")
        start_time = time.time()
        batch_results = rag.batch_retrieve(queries, top_k=3)
        batch_time = time.time() - start_time
        print(f"   批量检索耗时: {batch_time:.2f}秒")
        print(f"   平均每个查询: {batch_time/len(queries):.3f}秒")
        
        # 7. 性能对比
        print("\n" + "=" * 60)
        print("检索性能对比")
        print("=" * 60)
        speedup = individual_time / batch_time if batch_time > 0 else 0
        print(f"逐个检索: {individual_time:.2f}秒")
        print(f"批量检索: {batch_time:.2f}秒")
        print(f"性能提升: {speedup:.2f}x")
        print(f"节省时间: {individual_time - batch_time:.2f}秒 ({(1 - batch_time/individual_time)*100:.1f}%)")
        
        # 8. 测试生成性能（可选，因为会调用LLM API）
        print("\n8. 测试生成性能（可选）...")
        print("   注意：生成测试会调用LLM API，可能产生费用")
        
        user_input = input("   是否继续测试生成性能？(y/n): ")
        if user_input.lower() == 'y':
            # 准备上下文
            contexts_list = [result.contexts for result in batch_results[:5]]  # 只测试前5个
            test_queries = queries[:5]
            
            # 逐个生成
            print("\n   测试逐个生成...")
            start_time = time.time()
            individual_gen_results = []
            for query, contexts in zip(test_queries, contexts_list):
                result = rag.generate(query, contexts)
                individual_gen_results.append(result)
            individual_gen_time = time.time() - start_time
            print(f"   逐个生成耗时: {individual_gen_time:.2f}秒")
            
            # 批量生成
            print("\n   测试批量生成...")
            start_time = time.time()
            batch_gen_results = rag.batch_generate(test_queries, contexts_list)
            batch_gen_time = time.time() - start_time
            print(f"   批量生成耗时: {batch_gen_time:.2f}秒")
            
            # 生成性能对比
            print("\n" + "=" * 60)
            print("生成性能对比")
            print("=" * 60)
            gen_speedup = individual_gen_time / batch_gen_time if batch_gen_time > 0 else 0
            print(f"逐个生成: {individual_gen_time:.2f}秒")
            print(f"批量生成: {batch_gen_time:.2f}秒")
            print(f"性能提升: {gen_speedup:.2f}x")
            print(f"节省时间: {individual_gen_time - batch_gen_time:.2f}秒")
            
            # 显示一些生成结果
            print("\n" + "=" * 60)
            print("生成结果示例")
            print("=" * 60)
            for i, (query, result) in enumerate(zip(test_queries[:3], batch_gen_results[:3])):
                print(f"\n查询 {i+1}: {query[:50]}...")
                print(f"答案: {result.response[:100]}...")
        else:
            print("   跳过生成性能测试")
        
        # 9. 总结
        print("\n" + "=" * 60)
        print("总结")
        print("=" * 60)
        print(f"✓ 批量检索比逐个检索快 {speedup:.2f}x")
        print("✓ 批量处理特别适合大规模评测场景")
        print("✓ 建议在prepare_experiment_dataset时使用批量方法")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ 缺少依赖: {e}")
        print("\n请安装所需依赖：")
        print("  pip install faiss-cpu langchain langchain-openai")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

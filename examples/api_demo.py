"""API使用演示

展示如何通过Python客户端调用RAG Benchmark API
"""

import requests
import time
import json

API_BASE_URL = "http://localhost:8000"


def print_section(title):
    """打印分隔符"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_datasets():
    """演示数据集相关API"""
    print_section("1. 数据集管理")
    
    # 列出所有数据集
    print("📚 获取数据集列表...")
    response = requests.get(f"{API_BASE_URL}/datasets")
    datasets = response.json()
    print(f"可用数据集: {datasets}")
    
    if not datasets:
        print("⚠️  没有可用的数据集")
        return None
    
    # 获取第一个数据集的统计信息
    dataset_name = datasets[0]
    print(f"\n📊 获取 '{dataset_name}' 数据集统计...")
    response = requests.post(
        f"{API_BASE_URL}/datasets/stats",
        json={"name": dataset_name}
    )
    stats = response.json()
    print(f"记录数: {stats['record_count']}")
    print(f"语料库大小: {stats['corpus_count']}")
    print(f"平均问题长度: {stats['avg_input_length']:.1f}")
    
    # 获取样本
    print(f"\n📝 获取数据样本...")
    response = requests.post(
        f"{API_BASE_URL}/datasets/sample",
        json={"name": dataset_name},
        params={"n": 2}
    )
    samples = response.json()
    print(f"样本数: {samples['count']}")
    if samples['samples']:
        sample = samples['samples'][0]
        print(f"\n示例问题: {sample['user_input'][:100]}...")
        print(f"参考答案: {sample['reference'][:100]}...")
    
    return dataset_name


def demo_rag(dataset_name):
    """演示RAG系统相关API"""
    print_section("2. RAG系统管理")
    
    # 创建RAG实例
    rag_name = "demo_rag"
    print(f"🤖 创建RAG实例 '{rag_name}'...")
    response = requests.post(
        f"{API_BASE_URL}/rag/create",
        json={
            "name": rag_name,
            "model_name": "gpt-3.5-turbo",
            "embedding_model": "text-embedding-3-small",
            "config": {
                "top_k": 3,
                "temperature": 0.7
            }
        }
    )
    result = response.json()
    print(f"✅ {result['message']}")
    
    # 列出所有RAG实例
    print(f"\n📋 列出所有RAG实例...")
    response = requests.get(f"{API_BASE_URL}/rag/list")
    rags = response.json()
    print(f"RAG实例: {rags['rags']}")
    
    # 索引文档（示例）
    print(f"\n📚 索引示例文档...")
    documents = [
        "Python是一种高级编程语言，由Guido van Rossum创建。",
        "Python具有简洁的语法和强大的标准库。",
        "Python广泛应用于Web开发、数据科学、人工智能等领域。"
    ]
    response = requests.post(
        f"{API_BASE_URL}/rag/index",
        json={
            "rag_name": rag_name,
            "documents": documents
        }
    )
    result = response.json()
    print(f"✅ {result['message']}")
    
    # 测试查询
    print(f"\n🔍 测试查询...")
    query = "Python是什么？"
    response = requests.post(
        f"{API_BASE_URL}/rag/query",
        json={
            "rag_name": rag_name,
            "query": query
        }
    )
    result = response.json()
    print(f"问题: {result['query']}")
    print(f"答案: {result['answer']}")
    print(f"检索到 {len(result['contexts'])} 个上下文")
    
    return rag_name


def demo_evaluation(dataset_name, rag_name):
    """演示评测相关API"""
    print_section("3. 评测任务")
    
    # 启动评测
    print(f"🚀 启动评测任务...")
    response = requests.post(
        f"{API_BASE_URL}/evaluate/start",
        json={
            "dataset_name": dataset_name,
            "rag_name": rag_name,
            "eval_type": "e2e",
            "sample_size": 5
        }
    )
    result = response.json()
    task_id = result['task_id']
    print(f"✅ 任务已启动")
    print(f"任务ID: {task_id}")
    
    # 轮询任务状态
    print(f"\n⏳ 等待评测完成...")
    max_attempts = 60
    attempt = 0
    
    while attempt < max_attempts:
        response = requests.get(f"{API_BASE_URL}/evaluate/status/{task_id}")
        status = response.json()
        
        progress = int(status['progress'] * 100)
        print(f"\r进度: {progress}% | 状态: {status['status']}", end="", flush=True)
        
        if status['status'] == 'completed':
            print("\n✅ 评测完成！")
            print(f"\n📊 评测结果:")
            metrics = status['result']['metrics']
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
            break
        elif status['status'] == 'failed':
            print(f"\n❌ 评测失败: {status.get('error', 'Unknown error')}")
            break
        
        time.sleep(2)
        attempt += 1
    
    if attempt >= max_attempts:
        print("\n⚠️  评测超时")
    
    return task_id


def demo_results():
    """演示结果查看API"""
    print_section("4. 查看结果")
    
    # 列出所有任务
    print("📋 获取所有评测任务...")
    response = requests.get(f"{API_BASE_URL}/evaluate/tasks")
    tasks = response.json()
    
    completed_tasks = [t for t in tasks['tasks'] if t['status'] == 'completed']
    print(f"已完成任务数: {len(completed_tasks)}")
    
    if completed_tasks:
        print("\n最近的评测结果:")
        for i, task in enumerate(completed_tasks[-3:], 1):
            print(f"\n任务 {i}:")
            print(f"  ID: {task['task_id']}")
            print(f"  创建时间: {task['created_at']}")
            if task.get('result'):
                print(f"  评测类型: {task['result'].get('eval_type', 'N/A')}")
                print(f"  样本数: {task['result'].get('sample_count', 'N/A')}")


def main():
    """主函数"""
    print("\n" + "🎯" * 30)
    print("  RAG Benchmark API 演示")
    print("🎯" * 30)
    
    try:
        # 检查API是否可用
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print("❌ API服务不可用，请先启动API服务")
            print("运行: ./start_api.sh")
            return
        
        print("✅ API服务正常运行\n")
        
        # 运行演示
        dataset_name = demo_datasets()
        if not dataset_name:
            print("⚠️  无法继续演示，请先准备数据集")
            return
        
        rag_name = demo_rag(dataset_name)
        demo_evaluation(dataset_name, rag_name)
        demo_results()
        
        print("\n" + "=" * 60)
        print("  演示完成！")
        print("  访问 http://localhost:8000/docs 查看完整API文档")
        print("=" * 60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 无法连接到API服务")
        print("请确保API服务正在运行: ./start_api.sh")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

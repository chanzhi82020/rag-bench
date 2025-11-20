#!/usr/bin/env python
"""测试ragas实际返回的结果结构"""

import sys

from rag_benchmark.evaluate import F1AtK, MAPMetric

sys.path.insert(0, 'src')

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from datasets import Dataset

# 创建一个简单的测试数据集
samples = [
    SingleTurnSample(
        user_input="What is Python?",
        response="Python is a high-level programming language.",
        retrieved_contexts=["Python is a programming language known for its simplicity."],
        retrieved_context_ids=["1", "3", "2"],
        reference="Python is a programming language.",
        reference_context_ids=['1', '2']
    ),
SingleTurnSample(
        user_input="What is Python?",
        response="Python is a high-level programming language.",
        retrieved_contexts=["Python is a programming language known for its simplicity."],
        retrieved_context_ids=["1", "3", "2"],
        reference="Python is a programming language.",
        reference_context_ids=['1', '2']
    ),
    SingleTurnSample(
        user_input="What is AI?",
        response="AI stands for Artificial Intelligence.",
        retrieved_contexts=["Artificial Intelligence is the simulation of human intelligence."],
        retrieved_context_ids=["1", "3", "2"],
        reference="AI is Artificial Intelligence.",
        reference_context_ids=['1', '2']
    ),
]

# 创建EvaluationDataset
eval_dataset = EvaluationDataset(samples=samples)

print("=" * 80)
print("测试 ragas evaluate() 返回的结果结构")
print("=" * 80)

try:
    # 运行评测（使用最简单的指标）
    print("\n1. 运行评测...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[F1AtK(), MAPMetric()],
        show_progress=False,
    )

    print(f"\n2. 结果类型: {type(result)}")
    print(f"   类名: {result.__class__.__name__}")

    print(f"\n3. 结果属性:")
    for attr in dir(result):
        if not attr.startswith('_'):
            print(f"   - {attr}")

    print(f"\n4. scores 属性:")
    if hasattr(result, 'scores'):
        print(f"   类型: {type(result.scores)}")
        print(f"   长度: {len(result.scores) if hasattr(result.scores, '__len__') else 'N/A'}")
        if hasattr(result.scores, '__iter__'):
            print(f"   第一个元素类型: {type(result.scores[0]) if len(result.scores) > 0 else 'N/A'}")
            if len(result.scores) > 0:
                print(f"   第一个元素内容:")
                import json

                print(json.dumps(result.scores[0], indent=4, default=str))

    print(f"\n5. to_pandas() 方法:")
    df = result.to_pandas()
    print(f"   DataFrame 形状: {df.shape}")
    print(f"   列名: {df.columns.tolist()}")
    print(f"\n   列类型:")
    for col in df.columns:
        print(f"     {col}: {df[col].dtype}")

    print(f"\n   前2行数据:")
    print(df.head(2))

    print(f"\n6. 数值列:")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    print(f"   {numeric_cols}")

    print(f"\n7. 计算指标平均值:")
    metrics = {}
    for col in numeric_cols:
        metrics[col] = float(df[col].mean())
    print(f"   {metrics}")

    print(f"\n8. 从scores提取指标:")
    if hasattr(result, 'scores') and isinstance(result.scores, list) and len(result.scores) > 0:
        first = result.scores[0]
        print(f"   第一个样本的键: {list(first.keys())}")

        # 尝试提取数值指标
        score_metrics = {}
        for key in first.keys():
            values = [s.get(key) for s in result.scores if key in s]
            # 检查是否都是数值
            if values and all(isinstance(v, (int, float)) for v in values if v is not None):
                valid = [v for v in values if v is not None]
                if valid:
                    score_metrics[key] = sum(valid) / len(valid)

        print(f"   提取的指标: {score_metrics}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback

    traceback.print_exc()

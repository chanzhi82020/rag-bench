# 背景
构建一个RAG Benchmark框架，可以参考或集成ARES，BIER等branchmark框架，重点集成RAGAS评估开源框架

1.维护golden数据集的构建基准，比如规模，QA质量，领域性等

2.维护多套golden数据集，用于可以支持RAG端到端的评测

+ 核心数据：
    - `user_input`：由golden数据集提供
    - `ground_truth`：由golden数据集提供
        * `reference`：正确答案
        * `reference_contexts`：正确答案相关的上下文或证据(分片)
    - `corpus`：可以是golden数据集QA对应的文档列表或分片列表
    - `retrieved_contexts`：缺省，由被评测的rag系统给出，该rag系统基于corpus构建，并将`user_input`传入得到的相关检索内容即为`retrieved_contexts`
    - `response`：缺省，由被评测的rag系统给出，该rag系统基于corpus构建，并将`user_input`传入得到的答案即为`response`

> 上面golden数据集主要维护的字段为query，ground_truth，corpus，retrieval_contexts；而缺省的部分应当由用户去填充，同时框架内部可以内置一个rag来快速填充缺省类型并评测作为baseline
>

+ 相关内置数据集：
    - public：
        * hotpotqa based
        * nq based
    - private：
        * 智能客服(自己转换为上面的核心数据)

3.维护RAG的评测算法/任务：可以通过集成RAGAS实现

+ 端到端：
    - 需要的数据：`user_input`, `reference`,`retrieved_contexts`,`response`等
    - 需要的算法：context_recall，context_precision, failthfulness, accuracy等
+ 分阶段：
    - 检索阶段：
        * 需要的数据：`user_input`, `reference`,`retrieved_contexts`,`response`等
        * 需要的算法：context_recall，context_precision 等
        * 可拓展的数据：`retrieved_context_ids`，`reference_context_ids`
        * 可拓展的算法：f1_score, recall@k、precision@k、MRR、NDCG@k等
    - 生成阶段：
        * 需要的数据：`user_input`, `reference`,`reference_contexts`or `retrieved_contexts`,`response`等等
        * 需要的算法：failthfulness,  accuracy，groundness，coherence等
        * 可拓展的数据：`reference_tool_calls`, `reference_topics`
        * 可拓展的算法：topic_adherence,  tool_call_f1等

4.打通用户从获取golden数据集，填充实验数据集到拿到评测结果分析的整个流程

+ 需要提供examples，用户如何获取golden数据集，并填充
+ 需要支持评测结果的对比分析



---

# **MVP - 项目规划**
## **MVP 总体目标**
构建一个可用于 **RAG 端到端 & 分阶段评测** 的最小可用框架，具备以下能力：

1. **维护 Golden Dataset**（q,a,c + corpus filelist）
2. **prepare_experiment_dataset**：基于 golden dataset 和用户RAG系统，生成实验数据（填充 retrieved_contexts, response）
3. **evaluate**：端到端与分阶段评测（基于 RAGAS + 指标扩展接口）
4. **支持两种使用方式**（有 / 无自带 RAG）
5. **结果对比与分析能力**
6. **提供基础示例与 baseline RAG**

---

# 1. 架构模块拆分（MVP）
```plain
rag_benchmark/
├── datasets/
│   ├── data/
│   │   ├── hotpotqa/
│   │   │   ├── qac.jsonl     (user_input, reference, reference_contexts)
│   │   │   ├── corpus.jsonl
│   │   │   ├── metadata.jsonl
│   │   ├── nq/
│   │   ├── private_customer_service/
│   └── validators/     (数据质量/格式校验)
│   └── converters/     (数据转换脚本)
│   └── golden.py       (golden数据集操作句柄)
│
├── prepare/
│   ├── base_rag/  (框架内置 baseline RAG)
│   ├── prepare_experiment_dataset.py
│   ├── schema.py  (ExperimentDataset schema)
│
├── evaluate/
│   ├── metrics_ragas.py       (集成 RAGAS)
│   ├── metrics_retrieval.py   (recall@k / precision@k / mrr / ndcg)
│   ├── metrics_generation.py  (faithfulness / grounding / coherence)
│   ├── evaluator.py
│
├── analysis/
│   ├── compare.py   (模型对比)
│   ├── visualize.py (图表/数值)
│
├── examples/
│   ├── load_golden_dataset.ipynb
│   ├── prepare_experiment_dataset.ipynb
│   ├── evaluate_e2e.ipynb
│   ├── evaluate_retrieval.ipynb
│   ├── evaluate_generation.ipynb
│
└── README.md
```

---

# 2. 数据格式规范（MVP 版）

## **Golden Dataset构建基准**
### 数据质量标准
- **QA质量**：问题清晰无歧义，答案准确完整
- **上下文质量**：与问题高度相关，信息密度适中
- **规模基准**：小型(100-500条)、中型(1000-5000条)、大型(10000+条)
- **领域性**：支持通用领域和专业领域，具备领域代表性

## **Golden Dataset**
由框架内维护，不支持从文档自动构建。

### golden qac.jsonl
```plain
{
  "user_input": "…",
  "reference": "正确答案",
  "reference_contexts": ["段落1", "段落2", ...]
  "reference_contexts_ids": ["段落1_id", "段落2_id", ...] # optional
}
```

### corpus.jsonl
```plain
{
  "reference_context": "段落1",
  "reference_context_id": "段落1_id",
  "title": "段落所属文章标题"
}
```



---

## **Experiment Dataset (准备阶段输出)**
```plain
{
  "user_input": "...",
  "reference": "...",
  "reference_contexts": [...],
  "retrieved_contexts": [...],  # 用户提供 or baseline RAG 生成
  "response": "..."             # 用户提供 or baseline RAG 生成
}
```

---

# 3. MVP 功能任务拆解
## **A. 数据模块（Dataset）**
### **A1. Golden Dataset 结构定义**
+ 设计 golden dataset 目录规范
+ 定义 qac.jsonl 格式
+ 定义 corpus filelist 格式
+ 提供加载接口：`load_golden_dataset(name)`

### **A2. 内置 Golden 数据集准备**
+ hotpotQA（子集）
+ NQ（子集）
+ private: 智能客服（格式转换为 qac + corpus filelist）

---

## **B. 实验数据集准备阶段（Prepare）**
### **B1. ExperimentDataset schema 定义**
+ 直接集成RAGAS的RagasDataset

### **B2. prepare_experiment_dataset 实现**
提供两种方式：

#### **方式 1：用户提供 RAG**
```plain
exp = prepare_experiment_dataset(golden_ds, rag_system)
```

自动填充 retrieved_contexts & response。

#### **方式 2：无 RAG — 使用 baseline**
```plain
exp = prepare_experiment_dataset(golden_ds)
```

### **B3. 保存/加载实验数据集**
```plain
exp.save(path)
exp = load_experiment_dataset(path)
```

---

## **C. 评测阶段（Evaluate）**
### **C1. 端到端评测集成 RAGAS**
指标：

+ faithfulness
+ answer_relevance
+ context_relevance
+ answer_correctness / accuracy

### **C2. 检索阶段指标**
数据：user_input, reference_contexts, retrieved_contexts

+ context_recall
+ context_precision
+ recall@k / precision@k
+ mrr
+ ndcg@k

### **C3. 生成阶段指标**
数据：reference / retrieved_contexts / response

+ faithfulness
+ grounding
+ coherence
+ correctness

### **C4. 主评测接口**
```plain
results = evaluate(experiment_dataset, metrics={...})
```

---

## **D. 分析与对比（Analysis）**
### **D1. 同一模型不同版本对比**
+ 表格/图表：accuracy/faithfulness/precision@k 对比

### **D2. 不同模型之间对比**
+ 生成性能对比报告

### **D3. 回溯样例**
+ 找出 bad cases：faithfulness 低、retrieval recall 低等
+ 导出问题清单

---

## **E. baseline RAG（可选但建议）**
-faiss / chroma 作为检索器  
-小型开源 LLM（如 Qwen 1.5）作为基线生成  
允许用户快速对 golden 数据进行基线评测

---

## **F. 示例（Examples / Tutorials）**
1. 如何加载 golden dataset
2. 如何使用用户自己的 RAG 填充 retrieved_contexts 和 response
3. 如何执行 evaluate
4. 如何分析结果和对比多个模型

---

# 4. MVP 版本范围（Scope）
### **包含**
✔ 固定格式的 golden dataset（q,a,c + corpus-filelist）  
✔ prepare & evaluate 两阶段  
✔ RAGAS 集成  
✔ retrieval & generation metrics  
✔ baseline RAG  
✔ results compare

### **不包含（未来版本）**
✘ 从文档构建 golden dataset（doc → qac）  
✘ prompt-based data generation  
✘ web UI（可后续加入）  
✘ dataset 自动清洗与增强

---

# 5. 输出物汇总（Deliverables）
### **技术产出**
+ Python package（rag_benchmark）
+ 内置 golden datasets（hotpotQA / NQ / private）
+ baseline_rag 模块
+ metrics 模块 + evaluator
+ prepare_experiment_dataset 工具
+ evaluate 工具
+ examples.ipynb 系列示例

### **文档产出**
+ README（架构、使用方法）
+ Golden dataset 规范
+ Metrics 文档
+ Quickstart 教程（5min 上手）

---


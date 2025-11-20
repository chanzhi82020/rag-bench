# Dataset Setup Guide

由于数据集文件较大，它们不包含在Git仓库中。请按照以下步骤设置数据集。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/chanzhi82020/rag-bench.git
cd rag-bench
```

### 2. 下载数据集

数据集文件应放置在 `scripts/datasets/` 目录下。

#### 选项A：使用提供的数据集（如果有）

如果你有访问权限，可以从以下位置下载：
- 内部存储
- 云存储链接
- 其他共享位置

#### 选项B：从原始来源下载

**HotpotQA:**
```bash
# 下载并解压到 scripts/datasets/hotpotqa/
```

**XQuAD:**
```bash
# 下载并解压到 scripts/datasets/xquad/
```

**Natural Questions:**
```bash
# 下载大文件（1GB+）
wget https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz
mv simplified-nq-train.jsonl.gz scripts/datasets/v1.0-simplified_nq-dev-all.jsonl.gz
```

## 目录结构

```
scripts/datasets/
├── hotpotqa/
│   ├── distractor/
│   │   ├── corpus.jsonl
│   │   ├── metadata.json
│   │   └── qac.jsonl
│   └── fullwiki/
│       ├── corpus.jsonl
│       ├── metadata.json
│       └── qac.jsonl
├── xquad/
│   └── [language]/
│       ├── corpus.jsonl
│       ├── metadata.json
│       └── qac.jsonl
└── v1.0-simplified_nq-dev-all.jsonl.gz
```

## 存储需求

- **HotpotQA**: ~120 MB
- **XQuAD**: ~50 MB
- **Natural Questions**: ~1 GB

**总计**: ~1.2 GB

## 注意事项

1. 数据集文件已添加到 `.gitignore`，不会被提交到Git
2. 每个开发者需要独立下载数据集
3. 建议使用软链接或符号链接指向共享存储位置

## 验证安装

运行以下命令验证数据集是否正确安装：

```python
from rag_benchmark.datasets import GoldenDataset

# 测试加载数据集
ds = GoldenDataset("xquad", subset="zh")
print(f"Loaded {len(ds)} records")
```

## 故障排除

**问题**: 找不到数据集文件

**解决方案**:
1. 确认文件路径正确
2. 检查文件权限
3. 验证文件格式（JSONL）

**问题**: 文件太大无法下载

**解决方案**:
1. 使用断点续传工具（如 `wget -c` 或 `curl -C -`）
2. 考虑只下载需要的数据集子集
3. 联系团队获取内部存储访问权限

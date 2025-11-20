# RAG Benchmark 文档

## 文档导航

### 快速开始
- **[QUICKSTART.md](QUICKSTART.md)** - 5分钟快速上手指南
  - 环境安装
  - 启动服务
  - 基本使用流程
  - 常见问题

### API文档
- **[API.md](API.md)** - 完整的REST API接口文档
  - 模型仓库API
  - 数据集API
  - RAG系统API
  - 评测API
  - 使用示例

### 架构设计
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 系统架构和设计文档
  - 系统架构图
  - 核心模块设计
  - 数据流
  - 扩展点
  - 性能优化

## 模块文档

各模块的详细文档位于源码目录：

- `src/rag_benchmark/datasets/README.md` - 数据集模块
- `src/rag_benchmark/prepare/README.md` - RAG准备模块
- `src/rag_benchmark/evaluate/README.md` - 评测模块
- `src/rag_benchmark/analysis/README.md` - 分析模块
- `src/rag_benchmark/api/README.md` - API模块

## 示例代码

查看 `examples/` 目录获取完整示例：

- `load_dataset.py` - 加载数据集
- `prepare_experiment_dataset.py` - 准备实验数据
- `custom_rag_integration.py` - 自定义RAG集成
- `evaluate_rag_system.py` - 完整评测流程
- `compare_rag_systems.py` - 对比多个RAG系统
- `baseline_rag_example.py` - Baseline RAG使用
- `batch_processing_demo.py` - 批量处理示例
- `api_demo.py` - API使用演示

## 贡献指南

如果你想为文档做贡献：

1. 文档使用Markdown格式
2. 保持简洁清晰的结构
3. 提供实际可运行的示例
4. 更新相关的导航链接

## 反馈

如果文档有任何问题或建议，请：

- 提交Issue
- 发起Discussion
- 提交Pull Request

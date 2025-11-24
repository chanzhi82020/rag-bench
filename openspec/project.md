# Project Context

## Purpose
RAG Benchmark Framework 是一个用于评测RAG（Retrieval-Augmented Generation）系统性能的综合性Python框架。该项目集成了RAGAS评估框架，支持端到端和分阶段的RAG评测，旨在为RAG系统提供标准化的评估工具和基准测试平台。

主要目标：
- 提供标准化的RAG系统评测流程
- 支持多种公开数据集和自定义数据集
- 实现可视化的评测结果分析
- 提供灵活的RAG系统接口供用户集成

## Tech Stack

### 后端
- **Python 3.11+** - 主要编程语言
- **FastAPI** - Web API框架，提供自动化的API文档
- **Pydantic** - 数据验证和序列化
- **LangChain** - LLM集成框架
- **RAGAS** - RAG评估框架核心
- **FAISS** - 高效向量检索
- **uv** - 现代Python包管理工具
- **Uvicorn** - ASGI服务器

### 前端
- **React 18** - 现代化UI框架
- **TypeScript** - 类型安全的JavaScript超集
- **Vite** - 快速构建工具
- **Tailwind CSS** - 实用优先的CSS框架
- **Recharts** - 数据可视化图表库
- **Axios** - HTTP客户端
- **Headless UI** - 无样式组件库

### 开发工具
- **pytest** - 单元测试框架（覆盖率要求）
- **black** - 代码格式化（行长度88）
- **isort** - 导入排序（兼容black配置）
- **mypy** - 静态类型检查
- **hypothesis** - 属性测试
- **Git LFS** - 大文件版本控制

## Project Conventions

### 代码风格
- **Python**: 遵循PEP 8，使用black进行格式化（行长度88）
- **TypeScript**: 严格模式，使用ESLint和Prettier
- **命名规范**:
  - Python类使用PascalCase（如 `GoldenDataset`）
  - Python函数和变量使用snake_case（如 `prepare_experiment_dataset`）
  - TypeScript组件使用PascalCase，变量使用camelCase
- **类型注解**: Python代码强制要求类型注解，使用mypy检查
- **文档字符串**: 使用中文编写docstring，遵循Google风格

### 架构模式
- **模块化设计**: 按功能划分模块（datasets, prepare, evaluate, analysis, api）
- **抽象接口**: 使用ABC定义接口（如 `RAGInterface`）
- **数据类**: 大量使用 `@dataclass` 定义数据结构
- **注册模式**: 使用注册表管理数据集（`DATASET_REGISTRY`）
- **工厂模式**: 支持动态创建不同类型的加载器和转换器
- **批处理优化**: 支持批量检索和生成以提升性能
- **异步任务**: 支持长时间运行的评测任务，可断点续传

### 测试策略
- **单元测试**: 使用pytest，要求覆盖率报告
- **集成测试**: API端点的完整测试
- **属性测试**: 使用hypothesis进行边界测试
- **类型检查**: mypy强制检查，禁止未类型定义的代码
- **测试环境**: 独立的测试依赖配置（pyproject.toml中的[project.optional-dependencies.dev]）

### Git Workflow
- **分支策略**: 
  - `main` - 主分支，稳定版本
  - `develop` - 开发分支
  - `feature/*` - 功能分支
  - `refactor/*` - 重构分支
- **提交信息**: 使用中文，格式为"类型: 简短描述"
- **提交类型**: feat, fix, docs, style, refactor, test, chore
- **PR要求**: 必须通过所有CI检查（lint, test, type check）
- **Git Hooks**: 
  - post-commit: 自动运行Git LFS
  - post-checkout/post-merge: 同步大文件

## Domain Context

### RAG评估核心概念
- **Golden Dataset**: 包含问题、参考答案和参考上下文的标准化数据集
- **检索指标**: Recall@K, Precision@K, MRR, NDCG, Context Recall/Precision
- **生成指标**: Faithfulness（忠实度）, Answer Correctness（答案正确性）, Answer Relevancy（答案相关性）
- **评估模式**:
  - 端到端评估：完整RAG流程
  - 检索阶段评估：仅评估检索质量
  - 生成阶段评估：仅评估生成质量

### 支持的数据集
- **XQuAD**: 跨语言问答数据集，支持中文
- **HotpotQA**: 多跳问答数据集，需要推理
- **Natural Questions**: Google搜索真实问题
- **自定义数据集**: 支持JSONL格式导入

### RAG系统实现
- **BaselineRAG**: 基于FAISS + LLM的基础实现
- **DummyRAG**: 用于测试的简单实现
- **自定义RAG**: 通过实现`RAGInterface`接口集成

## Important Constraints

### 性能约束
- 批量处理优化，支持并行检索和生成
- 大数据集分块处理，避免内存溢出
- 评测任务支持异步执行和进度跟踪
- 向量索引使用FAISS进行高效检索

### 数据约束
- 数据集格式必须符合GoldenRecord/CorpusRecord模式
- 支持Git LFS管理大型数据文件
- 评测结果持久化存储，支持历史对比
- 模型配置统一管理，避免硬编码API密钥

### 安全约束
- API密钥通过环境变量或配置文件管理
- 不在代码中暴露敏感信息
- 支持代理配置，适应不同网络环境

## External Dependencies

### 核心依赖
- **datasets >= 4.4.1**: Hugging Face数据集库
- **ragas >= 0.3.9**: RAG评估框架
- **pydantic >= 2.0.0**: 数据验证
- **tqdm >= 4.64.0**: 进度条显示

### API依赖
- **fastapi >= 0.104.0**: Web框架
- **uvicorn[standard] >= 0.24.0**: ASGI服务器
- **python-multipart >= 0.0.6**: 文件上传支持
- **langchain-openai >= 0.0.5**: OpenAI集成

### 基准实现依赖
- **faiss-cpu >= 1.7.0**: 向量检索（可选GPU版本）
- **langchain >= 0.1.0**: LLM框架集成
- **matplotlib >= 3.5.0**: 图表绘制（分析模块）
- **numpy >= 1.21.0**: 数值计算
- **pandas >= 1.3.0**: 数据处理

### 镜像源
- 使用清华大学PyPI镜像源加速包安装
- index-url: https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
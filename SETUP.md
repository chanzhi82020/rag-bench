# RAG Benchmark 开发环境设置

本文档介绍如何使用conda创建开发环境。

## 前置要求

- 已安装 Anaconda 或 Miniconda

## 快速开始

### 1. 创建conda虚拟环境

```bash
# 创建名为rag-bench的虚拟环境，指定Python 3.11
conda create -n rag-bench python=3.11 -y

# 激活环境
conda activate rag-bench
```

### 2. 安装uv包管理器

```bash
# 在conda环境中安装uv
pip install uv
```

### 3. 使用uv安装项目依赖

```bash
# 同步所有依赖（包括开发依赖）
uv sync

# 或者只安装核心依赖
uv pip install -e .

# 安装开发依赖
uv pip install -e ".[dev]"
```

### 4. 验证安装

```bash
# 验证Python版本
python --version  # 应该显示 Python 3.11.x

# 验证uv安装
uv --version

# 测试导入
python -c "from rag_benchmark.datasets import GoldenDataset; print('✓ Import successful')"
```

## 详细说明

### 为什么使用conda + uv？

1. **conda**: 管理Python版本和系统级依赖
2. **uv**: 快速的Python包管理器，比pip更快
3. **组合优势**: conda提供隔离环境，uv提供快速安装

### 环境管理命令

```bash
# 激活环境
conda activate rag-bench

# 退出环境
conda deactivate

# 查看已安装的包
uv pip list

# 删除环境（如果需要重新开始）
conda deactivate
conda env remove -n rag-bench
```

### 依赖管理

```bash
# 添加新依赖到pyproject.toml后，同步环境
uv sync

# 更新所有依赖到最新版本
uv pip install --upgrade -e ".[dev]"

# 查看依赖树
uv pip tree
```

## 开发工作流

### 1. 运行测试

```bash
# 激活环境
conda activate rag-bench

# 运行简单测试
python test_prepare_simple.py

# 运行示例
python src/rag_benchmark/examples/prepare_experiment_dataset.py
```

### 2. 代码格式化

```bash
# 格式化代码
black src/

# 排序导入
isort src/

# 类型检查
mypy src/
```

### 3. 运行pytest测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_prepare.py

# 查看覆盖率
pytest --cov=rag_benchmark --cov-report=html
```

## 常见问题

### Q: conda create 很慢怎么办？

A: 配置conda使用国内镜像：

```bash
# 添加清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

### Q: uv安装包失败？

A: pyproject.toml已配置使用清华镜像，如果仍然失败：

```bash
# 手动指定镜像
uv pip install -e . --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
```

### Q: 如何在不同项目间切换？

A: 使用conda环境管理：

```bash
# 查看所有环境
conda env list

# 切换到其他环境
conda activate other-env

# 切换回rag-bench
conda activate rag-bench
```

### Q: 如何导出环境配置？

A: 

```bash
# 导出conda环境
conda env export > environment.yml

# 从配置文件创建环境
conda env create -f environment.yml
```

## IDE配置

### VS Code

1. 安装Python扩展
2. 选择解释器：`Ctrl+Shift+P` -> "Python: Select Interpreter"
3. 选择 `rag-bench` conda环境

### PyCharm

1. File -> Settings -> Project -> Python Interpreter
2. 点击齿轮图标 -> Add
3. 选择 "Conda Environment" -> "Existing environment"
4. 选择 `rag-bench` 环境

## 完整设置脚本

### Windows (PowerShell)

```powershell
# 创建并激活环境
conda create -n rag-bench python=3.11 -y
conda activate rag-bench

# 安装uv和依赖
pip install uv
uv sync

# 验证
python -c "from rag_benchmark.datasets import GoldenDataset; print('✓ Setup complete!')"
```

### Linux/macOS (Bash)

```bash
# 创建并激活环境
conda create -n rag-bench python=3.11 -y
conda activate rag-bench

# 安装uv和依赖
pip install uv
uv sync

# 验证
python -c "from rag_benchmark.datasets import GoldenDataset; print('✓ Setup complete!')"
```

## 下一步

环境设置完成后，可以：

1. 查看 `README.md` 了解项目概述
2. 查看 `src/rag_benchmark/prepare/README.md` 了解prepare模块
3. 运行示例代码学习使用方法
4. 开始开发新功能

## 故障排除

如果遇到问题：

1. 确保conda已正确安装：`conda --version`
2. 确保在正确的环境中：`conda info --envs`
3. 尝试重新创建环境：
   ```bash
   conda deactivate
   conda env remove -n rag-bench
   conda create -n rag-bench python=3.11 -y
   conda activate rag-bench
   pip install uv
   uv sync
   ```

## 参考资源

- [Conda文档](https://docs.conda.io/)
- [uv文档](https://github.com/astral-sh/uv)
- [项目README](README.md)

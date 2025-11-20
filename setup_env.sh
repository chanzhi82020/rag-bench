#!/bin/bash
# RAG Benchmark 环境快速设置脚本 (Linux/macOS)

echo "=========================================="
echo "RAG Benchmark 环境设置"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

echo "✓ 找到conda: $(conda --version)"

# 创建conda环境
echo ""
echo "创建conda环境 'rag-bench' (Python 3.11)..."
conda create -n rag-bench python=3.11 -y

if [ $? -ne 0 ]; then
    echo "❌ 创建conda环境失败"
    exit 1
fi

echo "✓ conda环境创建成功"

# 激活环境
echo ""
echo "激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rag-bench

if [ $? -ne 0 ]; then
    echo "❌ 激活环境失败"
    exit 1
fi

echo "✓ 环境已激活"

# 安装uv
echo ""
echo "安装uv包管理器..."
pip install uv

if [ $? -ne 0 ]; then
    echo "❌ 安装uv失败"
    exit 1
fi

echo "✓ uv安装成功: $(uv --version)"

# 同步依赖
echo ""
echo "同步项目依赖..."
uv sync

if [ $? -ne 0 ]; then
    echo "❌ 同步依赖失败"
    exit 1
fi

echo "✓ 依赖同步成功"

# 验证安装
echo ""
echo "验证安装..."
python -c "from rag_benchmark.prepare import ExperimentRecord; print('✓ prepare模块导入成功')"

if [ $? -ne 0 ]; then
    echo "❌ 验证失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 环境设置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 激活环境: conda activate rag-bench"
echo "  2. 运行测试: python test_prepare_simple.py"
echo "  3. 查看示例: python src/rag_benchmark/examples/prepare_experiment_dataset.py"
echo ""

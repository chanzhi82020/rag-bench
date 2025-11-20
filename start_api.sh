#!/bin/bash

# 启动RAG Benchmark API服务

echo "🚀 启动RAG Benchmark API服务..."
echo ""

# 检查uv
if ! command -v uv &> /dev/null; then
    echo "❌ 错误: 未找到uv"
    echo "请先安装uv: pip install uv"
    exit 1
fi

# 安装API依赖
echo "📦 安装依赖..."
uv pip install -e ".[api]" || {
    echo "❌ 依赖安装失败"
    exit 1
}

# 启动服务
echo ""
echo "✅ 准备就绪！"
echo "📡 API服务: http://localhost:8000"
echo "� API文档: "http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

uv run uvicorn rag_benchmark.api.main:app --reload --host 0.0.0.0 --port 8000

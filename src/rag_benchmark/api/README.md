# RAG Benchmark API

FastAPI服务，提供RAG系统评测的Web接口。

## 启动服务

```bash
# 方式1: 使用启动脚本
./start_api.sh

# 方式2: 使用uvicorn
uvicorn rag_benchmark.api.main:app --reload

# 方式3: 直接运行
python -m rag_benchmark.api.main
```

## API文档

启动服务后访问: http://localhost:8000/docs

## 依赖

API服务需要以下额外依赖：

```bash
pip install fastapi uvicorn[standard] langchain-openai
```

或者安装完整依赖：

```bash
pip install -e ".[baseline]"
```

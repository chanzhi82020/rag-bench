#!/bin/bash

# 启动RAG Benchmark前端

echo "启动RAG Benchmark前端..."

cd frontend

# 检查node_modules
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    npm install
fi

# 启动开发服务器
echo "前端服务启动在 http://localhost:3000"
npm run dev

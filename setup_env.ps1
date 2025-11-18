# RAG Benchmark 环境快速设置脚本 (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "RAG Benchmark 环境设置" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 检查conda是否安装
try {
    $condaVersion = conda --version 2>&1
    Write-Host "✓ 找到conda: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda" -ForegroundColor Red
    exit 1
}

# 创建conda环境
Write-Host ""
Write-Host "创建conda环境 'rag-bench' (Python 3.11)..." -ForegroundColor Yellow
conda create -n rag-bench python=3.11 -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 创建conda环境失败" -ForegroundColor Red
    exit 1
}

Write-Host "✓ conda环境创建成功" -ForegroundColor Green

# 激活环境
Write-Host ""
Write-Host "激活环境..." -ForegroundColor Yellow
conda activate rag-bench

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 激活环境失败" -ForegroundColor Red
    Write-Host "请手动运行: conda activate rag-bench" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ 环境已激活" -ForegroundColor Green

# 安装uv
Write-Host ""
Write-Host "安装uv包管理器..." -ForegroundColor Yellow
pip install uv

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 安装uv失败" -ForegroundColor Red
    exit 1
}

$uvVersion = uv --version
Write-Host "✓ uv安装成功: $uvVersion" -ForegroundColor Green

# 同步依赖
Write-Host ""
Write-Host "同步项目依赖..." -ForegroundColor Yellow
uv sync

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 同步依赖失败" -ForegroundColor Red
    exit 1
}

Write-Host "✓ 依赖同步成功" -ForegroundColor Green

# 验证安装
Write-Host ""
Write-Host "验证安装..." -ForegroundColor Yellow
python -c "from rag_benchmark.prepare import ExperimentRecord; print('✓ prepare模块导入成功')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 验证失败" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ 环境设置完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步:" -ForegroundColor Yellow
Write-Host "  1. 激活环境: conda activate rag-bench"
Write-Host "  2. 运行测试: python test_prepare_simple.py"
Write-Host "  3. 查看示例: python src/rag_benchmark/examples/prepare_experiment_dataset.py"
Write-Host ""

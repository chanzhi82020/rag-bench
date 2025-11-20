"""API基础测试

简单的测试脚本，验证API的基本功能
"""

import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    """测试健康检查"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ 健康检查通过")


def test_list_datasets():
    """测试列出数据集"""
    response = client.get("/datasets")
    assert response.status_code == 200
    datasets = response.json()
    assert isinstance(datasets, list)
    print(f"✅ 数据集列表: {datasets}")


def test_create_rag():
    """测试创建RAG实例"""
    response = client.post(
        "/rag/create",
        json={
            "name": "test_rag",
            "model_name": "gpt-3.5-turbo",
            "config": {
                "top_k": 3
            }
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "message" in result
    print(f"✅ 创建RAG: {result['message']}")


def test_list_rags():
    """测试列出RAG实例"""
    response = client.get("/rag/list")
    assert response.status_code == 200
    result = response.json()
    assert "rags" in result
    print(f"✅ RAG列表: {result['rags']}")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  RAG Benchmark API 基础测试")
    print("=" * 60 + "\n")
    
    try:
        test_health()
        test_list_datasets()
        test_create_rag()
        test_list_rags()
        
        print("\n" + "=" * 60)
        print("  ✅ 所有测试通过！")
        print("=" * 60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_tests()

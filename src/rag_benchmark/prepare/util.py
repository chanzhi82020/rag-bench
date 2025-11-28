from typing import Dict, Any

import jmespath

def map_response(mapping: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """使用 jmespath 提取目标字段"""
    result = {}
    # JMESPath 表达式（语法比 JSONPath 更简洁）
    # 批量提取（jmespath 自动处理空值）
    for key, expr in mapping.items():
        value = jmespath.search(expr, data)
        # 兼容多值场景（如果需要返回列表，去掉 | [0] 即可）
        result[key] = value if value is not None else None
    return result
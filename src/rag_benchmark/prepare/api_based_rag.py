import json
import logging
from typing import Dict, List, Any
from typing import Optional, Tuple

import httpx
from jinja2 import Template

from rag_benchmark.prepare.rag_interface import (
    RAGConfig,
    RAGInterface,
    RetrievalResult,
    GenerationResult,
)
from rag_benchmark.prepare.util import map_response

logger = logging.getLogger(__name__)



class APIBasedRAG(RAGInterface):
    """
    适用于任何通过 HTTP API 暴露的远程 RAG 系统的适配器

    支持：
    - 分别配置检索和生成接口
    - 自定义 headers / api_key / timeout
    - 使用 Jinja2 模板定义 request payload
    - 使用 field_mapping 把任意响应结构映射为标准 RetrievalResult / GenerationResult
    - 自动批量（如果远程支持）
    """

    def __init__(
            self,
            config: Optional[RAGConfig] = None,
            # ==================== 检索接口配置 ====================
            retrieval_endpoint: Optional[str] = None,
            retrieval_method: str = "POST",
            retrieval_headers: Optional[Dict[str, str]] = None,
            retrieval_api_key: Optional[str] = None,
            retrieval_payload_template: Optional[str] = None,
            retrieval_field_mapping: Optional[Dict[str, str]] = None,
            # ==================== 生成接口配置 ====================
            generation_endpoint: Optional[str] = None,
            generation_method: str = "POST",
            generation_headers: Optional[Dict[str, str]] = None,
            generation_api_key: Optional[str] = None,
            generation_payload_template: Optional[str] = None,
            generation_field_mapping: Optional[Dict[str, str]] = None,
            # ==================== 公共配置 ====================
            timeout: float = 120.0,
            verify_ssl: bool = False,
    ):
        super().__init__(config)

        # 公共 httpx 客户端
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            verify=verify_ssl,
        )

        # ==================== 检索配置 ====================
        self.retrieval_endpoint = retrieval_endpoint
        self.retrieval_method = retrieval_method.upper()
        self.retrieval_headers = retrieval_headers or {}
        if retrieval_api_key:
            self.retrieval_headers.setdefault("Authorization", f"Bearer {retrieval_api_key}")
        self.retrieval_payload_template = retrieval_payload_template
        self.retrieval_tmpl = (
            Template(retrieval_payload_template) if retrieval_payload_template else None
        )
        self.retrieval_mapping = retrieval_field_mapping or {}

        # ==================== 生成配置 ====================
        self.generation_endpoint = generation_endpoint or retrieval_endpoint  # 允许共用
        self.generation_method = generation_method.upper()
        self.generation_headers = generation_headers or self.retrieval_headers.copy()
        if generation_api_key:
            self.generation_headers.setdefault("Authorization", f"Bearer {generation_api_key}")
        self.generation_payload_template = generation_payload_template
        self.generation_tmpl = (
            Template(generation_payload_template) if generation_payload_template else None
        )
        self.generation_mapping = generation_field_mapping or {}

        # 验证至少有一个接口可用
        if not self.retrieval_endpoint and not self.generation_endpoint:
            raise ValueError("至少需要配置 retrieval_endpoint 或 generation_endpoint 之一")

    # ====================== 内部工具方法 ======================

    async def _request(
            self,
            endpoint: str,
            method: str,
            headers: Dict[str, str],
            payload: Dict,
    ) -> Dict:
        """统一请求封装"""
        response = await self.client.request(
            method=method,
            url=endpoint,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def render_payload(self, template: Optional[Template], query: str, contexts: Optional[List[str]] = None, **kwargs:Any) -> Dict:
        """使用 Jinja2 渲染 payload"""
        if not template:
            # 默认简单结构
            data: dict[str, Any] = {"query": query}
            if contexts is not None:
                data["contexts"] = contexts
            return data
        return json.loads(template.render(query=query, contexts=contexts or []))


    # ====================== 核心接口实现 ======================
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        if not self.retrieval_endpoint:
            raise NotImplementedError("Retrieval endpoint not configured")

        k = top_k if top_k is not None else self.config.top_k
        payload = self.render_payload(self.retrieval_tmpl, query)
        payload.setdefault("top_k", k)  # 很多服务需要显式传 top_k

        raw_resp = await self._request(
            endpoint=self.retrieval_endpoint,
            method=self.retrieval_method,
            headers=self.retrieval_headers,
            payload=payload,
        )

        mapped = map_response(self.retrieval_mapping, raw_resp)

        return RetrievalResult(
            contexts=mapped.get("contexts", []),
            context_ids=mapped.get("context_ids"),
            scores=mapped.get("scores"),
            metadata=mapped.get("metadata"),
        )

    async def generate(self, query: str, contexts: List[str]) -> GenerationResult:
        if not self.generation_endpoint:
            raise NotImplementedError("Generation endpoint not configured")

        payload = self.render_payload(self.generation_tmpl, query, contexts)

        raw_resp = await self._request(
            endpoint=self.generation_endpoint,
            method=self.generation_method,
            headers=self.generation_headers,
            payload=payload,
        )

        mapped = map_response(self.generation_mapping, raw_resp)

        return GenerationResult(
            response=mapped.get("response") or "",
            multi_responses=mapped.get("multi_responses"),
            confidence=mapped.get("confidence"),
            metadata=mapped.get("metadata"),
        )


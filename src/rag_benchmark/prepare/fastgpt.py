import asyncio
from typing import Optional, Dict, Any

from rag_benchmark.prepare.api_based_rag import APIBasedRAG
from rag_benchmark.prepare.rag_interface import RAGConfig


class FastGPTAPIRAG(APIBasedRAG):
    """
    专为 FastGPT /api/v2/chat/completions 接口定制的黑盒合一 RAG 适配器
    保持 retrieve 和 generate 分离评测能力
    """

    def __init__(
            self,
            endpoint: str = "http://14.116.240.74:31301/api/v2/chat/completions",
            app_id: str = "691148a011b2a0f3e9ec49ea",
            chat_id: Optional[str] = None,  # 可选，保持会话
            api_key: Optional[str] = None,  # FastGPT 如果有 key 就填
            top_k: int = 8,
            **kwargs,
    ):
        # 统一的 payload 模板（Jinja2）
        payload_template = """
        {
            "messages": [
                {
                    "role": "user",
                    "content": "{{ query }}"
                }
            ],
            "appId": "{{ app_id }}",
            "chatId": "{{ chat_id | default('') }}",
            "detail": true,
            "stream": false,
            "retainDatasetCite": true
        }
        """

        super().__init__(
            config=RAGConfig(top_k=top_k),

            # 共用同一个 endpoint（因为 FastGPT 没有分开）
            retrieval_endpoint=endpoint,
            generation_endpoint=endpoint,

            # 认证（如果你的 FastGPT 需要 key）
            retrieval_api_key=api_key,
            generation_api_key=api_key,

            # payload 模板
            retrieval_payload_template=payload_template,
            generation_payload_template=payload_template,  # 一样

            # 关键：field_mapping 把 FastGPT 的复杂结构映射出来
            retrieval_field_mapping={
                "contexts": "responseData[?moduleType=='datasetSearchNode'].quoteList[*].q[]",
                "context_ids": "responseData[?moduleType=='datasetSearchNode'].quoteList[*].id[]",
                "scores": "responseData[?moduleType=='datasetSearchNode'].quoteList[*].score[0].value[]",
            },
            generation_field_mapping={
                "response": "choices[0].message.content",  # 最终答案
            },
            **kwargs,
        )
        # 传递额外参数给模板
        self.template_vars = {
            "appId": app_id,
            "chatId": chat_id or ""
        }

    def render_payload(self, template: Optional[Any], query: str, contexts=None, **kwargs: Any) -> Dict:
        data = super().render_payload(template, query, contexts)
        # 把模板里用到的变量传进去
        data.update(self.template_vars or {})
        return data


# rag = FastGPTAPIRAG(chat_id='tAVWh9k0JXSkhHSqUGmXvM1g', api_key='openapi-djfiGnd3fkNrp8XxBOjRlaiDAiOs8XrTpAJkAOKV7zDBSa2ezKBR59zGZI1Lyj')
# retrieval_result, generate_result = asyncio.run(rag.retrieve_and_generate("如何报销",3))
# print(retrieval_result.to_dict())
# print(generate_result.to_dict())
# rag = APIBasedRAG(
#     config=RAGConfig(top_k=5),
#
#     # 检索接口（例如自研向量检索服务）
#     retrieval_endpoint="http://14.116.240.74:31301/api/v2/chat/completions",
#     retrieval_api_key="openapi-djfiGnd3fkNrp8XxBOjRlaiDAiOs8XrTpAJkAOKV7zDBSa2ezKBR59zGZI1Lyj",
#     retrieval_payload_template="""
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "{{query}}"
#             }
#         ],
#         "appId": "691148a011b2a0f3e9ec49ea",
#         "chatId": "tAVWh9k0JXSkhHSqUGmXvM1g",
#         "detail": true,
#         "stream": false,
#         "retainDatasetCite": true
#     }
#     """,
#     retrieval_field_mapping={
#         # 关键：使用 JSONPath 通配 + 过滤
#         "contexts": "responseData[?moduleType=='datasetSearchNode'].quoteList[*].q[]",
#         "context_ids": "responseData[?moduleType=='datasetSearchNode'].quoteList[*].id[]",
#         "scores": "responseData[?moduleType=='datasetSearchNode'].quoteList[*].score[0].value[]",
#     },
#
#     # 生成接口（例如自研或第三方 LLM 推理服务）
#     generation_endpoint="http://14.116.240.74:31301/api/v2/chat/completions",
#     generation_headers={"Content-Type": "application/json"},
#     generation_api_key="openapi-djfiGnd3fkNrp8XxBOjRlaiDAiOs8XrTpAJkAOKV7zDBSa2ezKBR59zGZI1Lyj",
#     generation_payload_template="""
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "{{query}}"
#             }
#         ],
#         "appId": "691148a011b2a0f3e9ec49ea",
#         "chatId": "tAVWh9k0JXSkhHSqUGmXvM1g",
#         "detail": true,
#         "stream": false,
#         "retainDatasetCite": true
#     }
#     """,
#     generation_field_mapping={
#         "response": "choices[0].message.content",
#     },
# )
#
# retrieval_result, generate_result = rag.retrieve_and_generate("如何报销",3)
# print(retrieval_result.to_dict())
# print(generate_result.to_dict())

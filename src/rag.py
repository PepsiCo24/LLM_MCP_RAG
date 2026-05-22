# ============================================================
# src/rag.py - RAG 模块（VectorStore + EmbeddingRetriever）
# ============================================================
"""极简 RAG 实现：向量存储 + 嵌入检索。

ER 图对应:
    VectorStore         — 内存向量存储（余弦相似度）
    EmbeddingRetriever  — 调用 embedding API + 检索
"""

import math
from typing import Any

from openai import OpenAI


# ============================================================
# VectorStore — 内存向量存储
# ============================================================

class VectorStoreItem:
    """向量存储条目。"""
    __slots__ = ("embedding", "document", "metadata")

    def __init__(self, embedding: list[float], document: str, metadata: dict | None = None):
        self.embedding = embedding
        self.document = document
        self.metadata = metadata or {}


class VectorStore:
    """内存向量存储，支持余弦相似度检索。

    对应 ER 图中 VectorStore。
    """

    def __init__(self):
        self._items: list[VectorStoreItem] = []

    def add_embedding(self, embedding: list[float], document: str, metadata: dict | None = None) -> None:
        """添加一个向量条目。"""
        self._items.append(VectorStoreItem(embedding, document, metadata))

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        """余弦相似度检索，返回 top_k 最相似条目。

        Returns:
            [{"document": str, "metadata": dict, "score": float}, ...]
        """
        if not self._items:
            return []

        scored = []
        for item in self._items:
            score = self._cosine_similarity(query_embedding, item.embedding)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, item in scored[:top_k]:
            results.append({
                "document": item.document,
                "metadata": item.metadata,
                "score": round(score, 4),
            })
        return results

    @property
    def size(self) -> int:
        return len(self._items)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ============================================================
# EmbeddingRetriever — 嵌入 + 检索
# ============================================================

class EmbeddingRetriever:
    """嵌入检索器 — 调用 embedding API 生成向量并检索。

    对应 ER 图中 EmbeddingRetriever。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "BAAI/bge-m3",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.vector_store = VectorStore()
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_document(self, document: str) -> list[float]:
        """将文档编码为向量并存入向量库。"""
        embedding = self._embed(document)
        self.vector_store.add_embedding(embedding, document)
        return embedding

    def embed_query(self, query: str) -> list[float]:
        """将查询编码为向量（不存入向量库）。"""
        return self._embed(query)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """检索与查询最相关的文档。

        Returns:
            [{"document": str, "metadata": dict, "score": float}, ...]
        """
        if self.vector_store.size == 0:
            return []
        query_embedding = self.embed_query(query)
        return self.vector_store.search(query_embedding, top_k)

    def _embed(self, text: str) -> list[float]:
        """调用 SiliconFlow Embedding API。"""
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
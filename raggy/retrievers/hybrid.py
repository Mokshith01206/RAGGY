"""
Raggy Hybrid Retriever.
High-precision retrieval engine combining dense vector search and sparse keyword matching
via Reciprocal Rank Fusion (RRF).
"""

import math
from typing import List, Dict, Any, Optional
from collections import defaultdict


class RaggyScoredNode:
    """Represents a retrieved document node with combined hybrid relevance score."""
    def __init__(self, node_id: str, text: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.text = text
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"RaggyScoredNode(id={self.node_id}, score={self.score:.4f}, text={self.text[:40]}...)"


class RaggyBM25Indexer:
    """Lightweight in-memory BM25 sparse indexer."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\w+", text.lower())

    def fit(self, docs: List[Dict[str, Any]]) -> None:
        """Fit BM25 statistics over a collection of documents [{'id': ..., 'text': ...}]."""
        self.docs = docs
        self.doc_lens = []
        total_len = 0
        self.doc_freqs = defaultdict(int)

        for doc in docs:
            tokens = set(self._tokenize(doc["text"]))
            doc_len = len(self._tokenize(doc["text"]))
            self.doc_lens.append(doc_len)
            total_len += doc_len
            for t in tokens:
                self.doc_freqs[t] += 1

        n_docs = len(docs)
        self.avg_doc_len = total_len / n_docs if n_docs > 0 else 1.0

        self.idf = {}
        for t, df in self.doc_freqs.items():
            self.idf[t] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Calculate BM25 scores for query."""
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc in enumerate(self.docs):
            doc_tokens = self._tokenize(doc["text"])
            doc_len = self.doc_lens[idx] if idx < len(self.doc_lens) else len(doc_tokens)
            score = 0.0

            token_counts = defaultdict(int)
            for t in doc_tokens:
                token_counts[t] += 1

            for qt in query_tokens:
                if qt not in self.idf:
                    continue
                tf = token_counts[qt]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += self.idf[qt] * (numerator / (denominator + 1e-9))

            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"id": d["id"], "text": d["text"], "score": s, "metadata": d.get("metadata", {})} for d, s in scores[:top_k]]


class RaggyHybridRetriever:
    """
    Raggy Hybrid Retriever.
    Fuses dense semantic results and BM25 sparse keyword scores using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        rrf_k: int = 60,
        similarity_top_k: int = 5,
        vector_retriever: Optional[Any] = None,
    ):
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.similarity_top_k = similarity_top_k
        self.vector_retriever = vector_retriever
        self.bm25_indexer = RaggyBM25Indexer()

    def set_corpus(self, docs: List[Dict[str, Any]]) -> None:
        """Register document corpus for sparse BM25 indexing."""
        self.bm25_indexer.fit(docs)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RaggyScoredNode]:
        """Perform hybrid retrieval using Reciprocal Rank Fusion."""
        k = top_k or self.similarity_top_k

        # 1. Sparse keyword retrieval
        sparse_hits = self.bm25_indexer.search(query, top_k=k * 2)

        # 2. Dense semantic retrieval (if vector retriever is configured)
        dense_hits = []
        if self.vector_retriever is not None and hasattr(self.vector_retriever, "retrieve"):
            try:
                raw_dense = self.vector_retriever.retrieve(query)
                for item in raw_dense:
                    node_id = getattr(item, "node_id", None) or getattr(getattr(item, "node", None), "node_id", "unknown")
                    text = getattr(item, "text", None) or getattr(getattr(item, "node", None), "text", "")
                    score = getattr(item, "score", 0.0) or 0.0
                    meta = getattr(item, "metadata", {}) or getattr(getattr(item, "node", None), "metadata", {})
                    dense_hits.append({"id": node_id, "text": text, "score": score, "metadata": meta})
            except Exception:
                pass

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = defaultdict(float)
        node_catalog: Dict[str, Dict[str, Any]] = {}

        # Fuse dense ranks
        for rank, hit in enumerate(dense_hits):
            nid = str(hit["id"])
            node_catalog[nid] = hit
            fused_scores[nid] += self.alpha * (1.0 / (self.rrf_k + rank + 1))

        # Fuse sparse ranks
        for rank, hit in enumerate(sparse_hits):
            nid = str(hit["id"])
            if nid not in node_catalog:
                node_catalog[nid] = hit
            fused_scores[nid] += (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank + 1))

        # Sort by fused score
        sorted_nodes = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for nid, f_score in sorted_nodes:
            info = node_catalog[nid]
            enriched_meta = dict(info.get("metadata", {}))
            enriched_meta["raggy_fusion_score"] = f_score
            results.append(
                RaggyScoredNode(
                    node_id=nid,
                    text=info["text"],
                    score=f_score,
                    metadata=enriched_meta,
                )
            )

        return results

"""
Raggy Fast Server.
Built-in HTTP REST API service for document ingestion and retrieval queries.
Requires zero external dependencies (uses standard library http.server).
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any
from raggy.config import config
from raggy.retrievers.hybrid import RaggyHybridRetriever
from raggy.agents.reflective import RaggyReflectiveAgent
from raggy.chunking.smart_chunker import RaggySmartChunker


class RaggyServiceState:
    """Singleton holding in-memory retriever, corpus, and agent state."""
    def __init__(self):
        self.chunker = RaggySmartChunker()
        self.retriever = RaggyHybridRetriever()
        self.agent = RaggyReflectiveAgent(self.retriever)
        self.corpus = []

    def ingest_text(self, text: str, doc_id: str = "doc_0"):
        chunks = self.chunker.chunk_text(text, doc_id=doc_id)
        for c in chunks:
            self.corpus.append({
                "id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
            })
        self.retriever.set_corpus(self.corpus)
        return len(chunks)


_STATE = RaggyServiceState()


class RaggyHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request handler for Raggy REST API."""

    def _send_json(self, status: int, data: Dict[str, Any]):
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ["/", "/health", "/api/v1/health"]:
            self._send_json(200, {
                "status": "healthy",
                "engine": "Raggy",
                "version": config.version,
                "total_chunks_indexed": len(_STATE.corpus),
            })
        elif path in ["/info", "/api/v1/info"]:
            self._send_json(200, {
                "app": config.app_name,
                "version": config.version,
                "default_llm": config.default_llm_model,
                "default_embed": config.default_embed_model,
                "hybrid_alpha": config.hybrid_alpha,
                "chunk_size": config.chunk_size,
            })
        else:
            self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {str(e)}"})
            return

        if path in ["/query", "/api/v1/query"]:
            query_str = payload.get("query", "").strip()
            if not query_str:
                self._send_json(400, {"error": "'query' field is required"})
                return

            res = _STATE.agent.query(query_str)
            self._send_json(200, {
                "query": query_str,
                "answer": res.response,
                "confidence": res.confidence_score,
                "reflections": res.reflection_count,
                "sources": [
                    {
                        "id": n.node_id,
                        "score": n.score,
                        "text": n.text,
                        "metadata": n.metadata,
                    }
                    for n in res.source_nodes
                ]
            })

        elif path in ["/ingest", "/api/v1/ingest"]:
            text = payload.get("text", "")
            doc_id = payload.get("doc_id", f"doc_{len(_STATE.corpus)}")
            if not text:
                self._send_json(400, {"error": "'text' field is required for ingestion"})
                return

            count = _STATE.ingest_text(text, doc_id=doc_id)
            self._send_json(200, {
                "message": f"Successfully ingested document '{doc_id}'",
                "chunks_created": count,
                "total_index_size": len(_STATE.corpus),
            })
        else:
            self._send_json(404, {"error": "Endpoint not found"})


def run_raggy_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the Raggy HTTP API server."""
    server = HTTPServer((host, port), RaggyHTTPRequestHandler)
    print(f"[*] Raggy Server running at http://{host}:{port}")
    print(f"[*] API Health: http://{host}:{port}/health")
    print(f"[*] API Query:  POST http://{host}:{port}/api/v1/query")
    print(f"[*] API Ingest: POST http://{host}:{port}/api/v1/ingest")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Raggy Server...")
        server.server_close()

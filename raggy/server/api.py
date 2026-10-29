#hello4: Raggy Fast Server & Web Application
#hello4: Built-in HTTP REST API and Interactive Web UI for document upload, extraction, and RAG chat
#hello4: Requires zero external web dependencies (uses standard library http.server)

import os
import json
import base64
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List
from raggy.config import config
from raggy.retrievers.hybrid import RaggyHybridRetriever
from raggy.agents.reflective import RaggyReflectiveAgent
from raggy.chunking.smart_chunker import RaggySmartChunker
from raggy.readers.extractor import RaggyDocumentExtractor


class RaggyServiceState:
    #hello4: Central in-memory state holding uploaded documents, index, and reflective agent
    def __init__(self):
        self.extractor = RaggyDocumentExtractor()
        self.retriever = RaggyHybridRetriever(alpha=config.hybrid_alpha)
        self.agent = RaggyReflectiveAgent(self.retriever)
        self.corpus: List[Dict[str, Any]] = []
        self.documents_info: List[Dict[str, Any]] = []

    def ingest_document(self, filename: str, content: Optional[str] = None, file_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        #hello4: Ingest document, extract text, chunk and index
        extracted = self.extractor.extract_and_chunk(filename, file_bytes=file_bytes)
        chunks = extracted["chunks"]

        for c in chunks:
            self.corpus.append({
                "id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
            })

        self.retriever.set_corpus(self.corpus)

        doc_summary = {
            "filename": extracted["filename"],
            "format": extracted["format"],
            "page_count": extracted["page_count"],
            "total_chars": extracted["total_chars"],
            "chunk_count": extracted["chunk_count"],
            "preview": extracted["text"][:300] + ("..." if len(extracted["text"]) > 300 else ""),
        }
        self.documents_info.append(doc_summary)
        return doc_summary


# Type alias helper
Union_Content = Optional[str]

#hello4: Global state singleton
_STATE = RaggyServiceState()


#hello4: Embedded HTML Web Application Interface
RAGGY_HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Raggy - Document Upload & RAG Studio</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #0f172a; color: #f8fafc; display: flex; height: 100vh; overflow: hidden; }
    .sidebar { width: 380px; background: #1e293b; border-right: 1px solid #334155; display: flex; flex-direction: column; padding: 20px; }
    .logo { font-size: 1.4rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
    .subtitle { font-size: 0.85rem; color: #94a3b8; margin-bottom: 20px; }
    .dropzone { border: 2px dashed #475569; border-radius: 12px; padding: 24px; text-align: center; cursor: pointer; transition: 0.2s; background: #0f172a; }
    .dropzone:hover { border-color: #38bdf8; background: #1e293b; }
    .dropzone p { font-size: 0.9rem; color: #cbd5e1; margin-top: 8px; }
    .dropzone span { font-size: 0.75rem; color: #64748b; }
    input[type="file"] { display: none; }
    .doc-list { margin-top: 20px; flex: 1; overflow-y: auto; }
    .doc-list h4 { font-size: 0.8rem; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
    .doc-card { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .doc-card .name { font-weight: 600; font-size: 0.9rem; color: #f1f5f9; }
    .doc-card .stats { font-size: 0.75rem; color: #38bdf8; margin-top: 4px; }
    .doc-card .preview { font-size: 0.75rem; color: #94a3b8; margin-top: 6px; max-height: 60px; overflow: hidden; }
    .main { flex: 1; display: flex; flex-direction: column; height: 100vh; }
    .header { padding: 16px 24px; border-bottom: 1px solid #334155; background: #1e293b; display: flex; justify-content: space-between; align-items: center; }
    .status-badge { background: #0284c7; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; }
    .chat-container { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
    .message { max-width: 80%; padding: 14px 18px; border-radius: 12px; font-size: 0.95rem; line-height: 1.5; }
    .message.user { align-self: flex-end; background: #2563eb; color: white; }
    .message.assistant { align-self: flex-start; background: #1e293b; border: 1px solid #334155; color: #e2e8f0; }
    .message .meta { font-size: 0.75rem; color: #38bdf8; margin-top: 8px; border-top: 1px solid #334155; padding-top: 6px; }
    .input-bar { padding: 16px 24px; background: #1e293b; border-top: 1px solid #334155; display: flex; gap: 12px; }
    .input-bar input { flex: 1; background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; color: white; font-size: 0.95rem; outline: none; }
    .input-bar input:focus { border-color: #38bdf8; }
    .input-bar button { background: #0284c7; border: none; color: white; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    .input-bar button:hover { background: #0369a1; }
  </style>
</head>
<body>
  <div class="sidebar">
    <div class="logo">⚡ Raggy Studio</div>
    <div class="subtitle">Next-Gen Document RAG & Text Extraction</div>

    <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
      <div style="font-size: 2rem;">📄</div>
      <p><strong>Click to upload</strong> or drag & drop</p>
      <span>Supports PDF, TXT, MD, JSON, CSV</span>
      <input type="file" id="fileInput" accept=".pdf,.txt,.md,.json,.csv,.py" onchange="uploadFile(this.files[0])">
    </div>

    <div class="doc-list">
      <h4>Indexed Documents (<span id="docCount">0</span>)</h4>
      <div id="docsContainer">
        <p style="font-size: 0.8rem; color: #64748b;">No documents uploaded yet. Upload a PDF or TXT to start asking questions!</p>
      </div>
    </div>
  </div>

  <div class="main">
    <div class="header">
      <div>
        <h3 style="font-size: 1.1rem; font-weight: 600;">Document Q&A Assistant</h3>
        <span style="font-size: 0.8rem; color: #94a3b8;">Hybrid RRF Search + Reflective Verification</span>
      </div>
      <span class="status-badge" id="indexStatus">Index: Ready</span>
    </div>

    <div class="chat-container" id="chat">
      <div class="message assistant">
        👋 Welcome to <strong>Raggy</strong>! Upload any document (PDF, TXT, MD) using the panel on the left, and I will extract the text, chunk it hierarchically, and answer your questions with precise citations.
      </div>
    </div>

    <div class="input-bar">
      <input type="text" id="queryInput" placeholder="Ask a question about your uploaded document..." onkeydown="if(event.key==='Enter') sendQuery()">
      <button onclick="sendQuery()">Ask Raggy</button>
    </div>
  </div>

  <script>
    async function uploadFile(file) {
      if (!file) return;
      const dropzone = document.getElementById('dropzone');
      dropzone.style.opacity = '0.5';

      const reader = new FileReader();
      reader.onload = async function(e) {
        const base64Data = e.target.result.split(',')[1];
        try {
          const res = await fetch('/api/v1/upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: jsonPayload(file.name, base64Data)
          });
          const data = await res.json();
          dropzone.style.opacity = '1';
          if (data.error) {
            alert('Upload error: ' + data.error);
          } else {
            renderDoc(data);
            addMessage('assistant', `✅ Successfully extracted and indexed **${data.filename}** (${data.chunk_count} chunks, ${data.total_chars} characters). You can now ask questions about it!`);
          }
        } catch(err) {
          dropzone.style.opacity = '1';
          alert('Upload failed: ' + err);
        }
      };
      reader.readAsDataURL(file);
    }

    function jsonPayload(name, b64) {
      return JSON.stringify({ filename: name, data_base64: b64 });
    }

    function renderDoc(doc) {
      const container = document.getElementById('docsContainer');
      if (document.getElementById('docCount').innerText === '0') {
        container.innerHTML = '';
      }
      const count = parseInt(document.getElementById('docCount').innerText) + 1;
      document.getElementById('docCount').innerText = count;

      const card = document.createElement('div');
      card.className = 'doc-card';
      card.innerHTML = `
        <div class="name">📄 ${doc.filename}</div>
        <div class="stats">${doc.format.toUpperCase()} • ${doc.chunk_count} chunks • ${doc.total_chars} chars</div>
        <div class="preview">${doc.preview}</div>
      `;
      container.prepend(card);
      document.getElementById('indexStatus').innerText = `Index: ${count} Docs`;
    }

    async function sendQuery() {
      const input = document.getElementById('queryInput');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';

      addMessage('user', text);

      try {
        const res = await fetch('/api/v1/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: text })
        });
        const data = await res.json();
        
        let metaHtml = `Confidence: ${(data.confidence * 100).toFixed(0)}% | Reflections: ${data.reflections}`;
        if (data.sources && data.sources.length > 0) {
          metaHtml += '<br><strong>Top Source:</strong> ' + (data.sources[0].metadata.breadcrumb || data.sources[0].metadata.doc_id || 'Document');
        }

        addMessage('assistant', data.answer, metaHtml);
      } catch (err) {
        addMessage('assistant', 'Error querying index: ' + err);
      }
    }

    function addMessage(sender, text, meta) {
      const chat = document.getElementById('chat');
      const msg = document.createElement('div');
      msg.className = `message ${sender}`;
      msg.innerHTML = text.replace(/\\n/g, '<br>');
      if (meta) {
        const m = document.createElement('div');
        m.className = 'meta';
        m.innerHTML = meta;
        msg.appendChild(m);
      }
      chat.appendChild(msg);
      chat.scrollTop = chat.scrollHeight;
    }
  </script>
</body>
</html>
"""


class RaggyHTTPRequestHandler(BaseHTTPRequestHandler):
    #hello4: HTTP Request handler for JSON REST API and Web Interface

    def _send_html(self, html_content: str):
        #hello4: Send HTML response
        content_bytes = html_content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content_bytes)))
        self.end_headers()
        self.wfile.write(content_bytes)

    def _send_json(self, status: int, data: Dict[str, Any]):
        #hello4: Send JSON response
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
        #hello4: Handle CORS preflight
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        #hello4: Handle GET requests
        path = self.path.split("?")[0]
        if path in ["/", "/app", "/index.html"]:
            self._send_html(RAGGY_HTML_UI)
        elif path in ["/health", "/api/v1/health"]:
            self._send_json(200, {
                "status": "healthy",
                "engine": "Raggy",
                "version": config.version,
                "total_documents": len(_STATE.documents_info),
                "total_chunks_indexed": len(_STATE.corpus),
            })
        elif path in ["/documents", "/api/v1/documents"]:
            self._send_json(200, {
                "documents": _STATE.documents_info,
                "total_chunks": len(_STATE.corpus),
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
        #hello4: Handle POST endpoints
        path = self.path.split("?")[0]
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {str(e)}"})
            return

        # Upload and extract document
        if path in ["/upload", "/api/v1/upload"]:
            filename = payload.get("filename", "document.txt")
            b64_data = payload.get("data_base64", None)
            raw_text = payload.get("text", None)

            try:
                if b64_data:
                    file_bytes = base64.b64decode(b64_data)
                    summary = _STATE.ingest_document(filename, file_bytes=file_bytes)
                elif raw_text:
                    summary = _STATE.ingest_document(filename, file_bytes=raw_text.encode("utf-8"))
                else:
                    self._send_json(400, {"error": "Either 'data_base64' or 'text' is required."})
                    return

                self._send_json(200, summary)
            except Exception as e:
                self._send_json(500, {"error": f"Extraction error: {str(e)}"})

        # Query document index
        elif path in ["/query", "/api/v1/query"]:
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
        else:
            self._send_json(404, {"error": "Endpoint not found"})


def run_raggy_server(host: str = "0.0.0.0", port: int = 8000):
    #hello4: Start the Raggy HTTP API and Web Studio server
    server = HTTPServer((host, port), RaggyHTTPRequestHandler)
    print(f"[*] Raggy Web Studio & API running at http://localhost:{port}")
    print(f"[*] Open http://localhost:{port} in your browser to upload documents and chat!")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Raggy Server...")
        server.server_close()

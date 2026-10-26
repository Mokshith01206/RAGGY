# 🦙⚡ Raggy

> **The Next-Generation Modular Retrieval-Augmented Generation (RAG) & Agent Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://python.org)
[![Framework: Raggy](https://img.shields.io/badge/Framework-Raggy-orange.svg)]()

**Raggy** is a modernized, production-ready RAG framework designed for deep document understanding, multi-hop reasoning, and reliable context grounding. Built on top of modular indexing primitives, Raggy introduces native **Reciprocal Rank Fusion (RRF) Hybrid Search**, **Hierarchical Context-Preserving Chunking**, **Self-Reflective Agents**, and a **Built-in REST API Server & CLI**.

---

## 🌟 Why Raggy? (Custom Features)

Unlike standard retrieval pipelines, Raggy provides specialized out-of-the-box enhancements:

| Feature | Standard RAG | **Raggy** |
| :--- | :--- | :--- |
| **Hybrid Search** | External Vector DB required | **Native In-Memory BM25 + Dense RRF Fusion** with customizable alpha weights |
| **Document Chunking** | Blind fixed-token splitting | **`RaggySmartChunker`**: Hierarchical header breadcrumbs + metadata tracking |
| **Agent Reasoning** | Single-pass generation | **`RaggyReflectiveAgent`**: Confidence scoring and automated query reformulation |
| **Deployment** | Boilerplate code required | **`raggy serve`**: Zero-dependency REST API server built right in |
| **Interactive REPL** | Custom scripts needed | **`raggy chat`**: Instant command-line conversation over documents |

---

## 🚀 Quick Start

### 1. Installation

From the project root directory:

```bash
# Editable install
pip install -e .

# With optional web and LLM dependencies
pip install -e ".[all]"
```

---

### 2. Python API Usage

#### Hierarchical Smart Chunking
```python
from raggy import RaggySmartChunker

markdown_doc = """
# Project Apollo
## Overview
Apollo was the NASA program that landed the first humans on the Moon in 1969.

## Mission Accomplishments
The Apollo 11 mission crew consisted of Neil Armstrong, Buzz Aldrin, and Michael Collins.
"""

chunker = RaggySmartChunker(chunk_size=256, include_breadcrumb_prefix=True)
chunks = chunker.chunk_text(markdown_doc, doc_id="apollo_doc")

for chunk in chunks:
    print(f"ID: {chunk.chunk_id}")
    print(f"Breadcrumb: {chunk.metadata['breadcrumb']}")
    print(f"Content: {chunk.text}\n")
```

#### Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
```python
from raggy.retrievers import RaggyHybridRetriever

retriever = RaggyHybridRetriever(alpha=0.6, similarity_top_k=3)

# Load your document corpus
corpus = [
    {"id": "doc_1", "text": "Raggy is an advanced RAG framework with dynamic RRF."},
    {"id": "doc_2", "text": "LlamaIndex and LangChain are popular LLM libraries."},
    {"id": "doc_3", "text": "BM25 provides exact keyword matching while embeddings capture semantics."},
]
retriever.set_corpus(corpus)

# Hybrid search
results = retriever.retrieve("How does Raggy handle keywords and semantic search?")
for node in results:
    print(f"[{node.score:.4f}] {node.text}")
```

#### Self-Reflective Agent
```python
from raggy.retrievers import RaggyHybridRetriever
from raggy.agents import RaggyReflectiveAgent

retriever = RaggyHybridRetriever(alpha=0.5)
retriever.set_corpus(corpus)

agent = RaggyReflectiveAgent(
    retriever=retriever,
    confidence_threshold=0.65,
    max_iterations=2
)

response = agent.query("What makes Raggy hybrid search unique?")
print(f"Confidence: {response.confidence_score:.2f}")
print(f"Reflections: {response.reflection_count}")
print(f"Answer: {response.response}")
```

---

## 💻 Command Line Interface (CLI)

Raggy comes with a built-in CLI:

```bash
# Show framework info & active configuration
raggy info

# Launch interactive terminal chat over your documents
raggy chat --docs ./my_documents

# Start the REST API server on port 8000
raggy serve --port 8000

# Execute a one-shot query
raggy query "Explain the architecture of Raggy"
```

---

## 📡 Built-In REST API

Start the server using `raggy serve --port 8000`, then interact via HTTP:

* **Health Check**: `GET http://localhost:8000/health`
* **Ingest Data**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/ingest \
       -H "Content-Type: application/json" \
       -d '{"doc_id": "notes", "text": "Raggy makes retrieval seamless and fast."}'
  ```
* **Query Documents**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/query \
       -H "Content-Type: application/json" \
       -d '{"query": "What does Raggy do?"}'
  ```

---

## 🛠️ Uploading to Your Own GitHub Repository

This repository is initialized as a fresh Git repository ready for your own GitHub account:

```bash
# 1. Create a new empty repository on https://github.com/new named 'raggy'

# 2. Add your GitHub remote
git remote add origin https://github.com/<YOUR-USERNAME>/raggy.git

# 3. Commit all files
git add .
git commit -m "feat: initial commit for Raggy framework"

# 4. Push to your repo
git branch -M main
git push -u origin main
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

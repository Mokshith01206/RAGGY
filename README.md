# Raggy

The Next-Generation Modular Retrieval-Augmented Generation (RAG) and Agent Framework.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Repository](https://img.shields.io/badge/GitHub-Mokshith01206%2FRAGGY-blue.svg)](https://github.com/Mokshith01206/RAGGY)

Raggy is a production-grade, modular RAG framework engineered for high-precision document extraction, multi-hop retrieval, and verified generation. Built on top of extensible primitives, Raggy provides native Reciprocal Rank Fusion (RRF) hybrid search, hierarchical context-preserving document chunking, self-reflective agents, and an embedded zero-dependency Web Studio interface.

---

## Architectural Highlights

| Component | Standard Approach | Raggy Implementation |
| :--- | :--- | :--- |
| **Document Ingestion** | Blind plain-text parsing | **`RaggyDocumentExtractor`**: Native multi-format parser for PDF, TXT, MD, CSV, and JSON with automated page numbering and metadata normalization |
| **Hierarchical Chunking** | Fixed character/token windowing | **`RaggySmartChunker`**: Preserves document structure, section headers, and hierarchical breadcrumb trails across chunks |
| **Hybrid Retrieval** | Requires external vector database setup | **`RaggyHybridRetriever`**: Native in-memory BM25 lexical search combined with dense semantic vectors using Reciprocal Rank Fusion (RRF) |
| **Agent Reasoning** | Single-pass generation | **`RaggyReflectiveAgent`**: Evaluates context confidence scores and automatically reformulates queries on low-confidence hits |
| **Web Interface** | Third-party dashboard required | **`Raggy Web Studio`**: Embedded browser interface with drag-and-drop document upload and live chat |
| **CLI Tools** | Ad-hoc execution scripts | **`raggy` CLI**: Unified commands for serve, chat, upload, query, and inspection |

---

## Installation

### From Source

```bash
git clone https://github.com/Mokshith01206/RAGGY.git
cd RAGGY
pip install -e .
```

### With Optional Dependencies

```bash
pip install -e ".[all]"
```

---

## Quick Start

### 1. Document Upload and Text Extraction

Raggy extracts text from PDF and text documents, splits them into context-aware chunks, and initiates grounded Q&A:

```bash
# Ingest any document and query via CLI
python upload_doc.py "path/to/document.pdf"
```

Or using the CLI:

```bash
python -m raggy.cli upload "path/to/document.pdf"
```

### 2. Embedded Web Studio

Start the integrated web server (requires no external server dependencies):

```bash
python -m raggy.cli serve --port 8000
```

Open `http://localhost:8000` in any web browser to access the graphical document uploader and interactive chat interface.

### 3. Interactive Terminal Chat

```bash
python -m raggy.cli chat --docs "./my_documents"
```

---

## Python API Reference

### Document Extraction and Smart Chunking

```python
from raggy import RaggyDocumentExtractor

extractor = RaggyDocumentExtractor(chunk_size=512, chunk_overlap=50)

# Extract and hierarchically chunk a document (PDF, TXT, MD, JSON, CSV)
result = extractor.extract_and_chunk("annual_report.pdf")

print(f"Format: {result['format']}")
print(f"Pages: {result['page_count']}")
print(f"Total Characters: {result['total_chars']}")
print(f"Generated Chunks: {result['chunk_count']}")

for chunk in result["chunks"][:3]:
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Breadcrumb: {chunk.metadata.get('breadcrumb')}")
    print(f"Content Preview: {chunk.text[:120]}...\n")
```

### Reciprocal Rank Fusion (RRF) Hybrid Retrieval

```python
from raggy import RaggyHybridRetriever

retriever = RaggyHybridRetriever(
    alpha=0.6,             # Dense vs sparse weight balance
    rrf_k=60,              # Reciprocal Rank Fusion constant
    similarity_top_k=5     # Number of returned hits
)

# Index structured corpus
corpus = [
    {"id": "doc_1", "text": "Raggy implements hybrid search fusing BM25 and dense embeddings with RRF."},
    {"id": "doc_2", "text": "Reciprocal Rank Fusion standardizes rank scores across heterogeneous search spaces."},
    {"id": "doc_3", "text": "Self-reflective agents verify context alignment prior to answer synthesis."},
]
retriever.set_corpus(corpus)

# Execute hybrid retrieval
hits = retriever.retrieve("How does Raggy fuse search results?")
for hit in hits:
    print(f"[Score: {hit.score:.4f}] {hit.text}")
```

### Self-Reflective Retrieval Agent

```python
from raggy import RaggyReflectiveAgent, RaggyHybridRetriever

retriever = RaggyHybridRetriever(alpha=0.5)
retriever.set_corpus(corpus)

agent = RaggyReflectiveAgent(
    retriever=retriever,
    confidence_threshold=0.65,
    max_iterations=2
)

response = agent.query("Explain Reciprocal Rank Fusion in Raggy")
print(f"Confidence Score: {response.confidence_score:.2f}")
print(f"Reflection Iterations: {response.reflection_count}")
print(f"Answer: {response.response}")
```

---

## REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web Studio user interface |
| `GET` | `/api/v1/health` | System health check and total indexed document count |
| `GET` | `/api/v1/documents` | Catalog of uploaded documents with extraction statistics |
| `POST` | `/api/v1/upload` | Ingest base64-encoded or raw text documents for parsing and indexing |
| `POST` | `/api/v1/query` | Execute reflective retrieval query against the active index |

---

## Repository Structure

```text
RAGGY/
|-- raggy/                         # Core Raggy package
|   |-- __init__.py                # Top-level framework exports
|   |-- cli.py                     # CLI tool (serve, chat, upload, query)
|   |-- config.py                  # Global settings, model parameters, and thresholds
|   |-- readers/                   # Universal document extraction (PDF, TXT, MD)
|   |   |-- __init__.py
|   |   `-- extractor.py
|   |-- chunking/                  # Context-preserving hierarchical chunking
|   |   |-- __init__.py
|   |   `-- smart_chunker.py
|   |-- retrievers/                # BM25 + dense RRF hybrid retrieval
|   |   |-- __init__.py
|   |   `-- hybrid.py
|   |-- agents/                    # Self-reflective retrieval agents
|   |   |-- __init__.py
|   |   `-- reflective.py
|   `-- server/                    # Embedded HTTP API and Web Studio
|       |-- __init__.py
|       `-- api.py
|-- raggy-core/                    # Core indexing primitives
|-- raggy-integrations/            # Integration connectors
|-- raggy-instrumentation/         # Telemetry and instrumentation
|-- raggy-utils/                   # Utility packages
|-- raggy-dev/                     # Developer tools and test runners
|-- demo_raggy.py                  # Verification and benchmark runner
|-- upload_doc.py                  # Standalone document ingestion and Q&A script
|-- pyproject.toml                 # Package specification and metadata
`-- README.md                      # Documentation
```

---

## Contributing

Contributions are welcome. Please ensure new features include corresponding unit tests and documentation updates.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-capability`).
3. Commit your changes (`git commit -m "feat: add new capability"`).
4. Push to the branch (`git push origin feature/new-capability`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

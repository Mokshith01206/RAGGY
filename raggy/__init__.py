"""
🦙⚡ Raggy: The Next-Generation Modular RAG & Agent Framework.

A modernized, high-performance retrieval-augmented generation engine
featuring dynamic Reciprocal Rank Fusion hybrid search, context-preserving
hierarchical chunking, and self-reflective agents.
"""

import sys
import os
from pathlib import Path

# Add core engine to path dynamically if running locally
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CORE_PATH = _REPO_ROOT / "llama-index-core"
if _CORE_PATH.exists() and str(_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(_CORE_PATH))

# Raggy version
__version__ = "1.0.0"

# Custom Raggy innovations
from raggy.config import config, RaggyConfig
from raggy.chunking.smart_chunker import (
    RaggyChunk,
    RaggySmartChunker,
    smart_chunk_document,
)
from raggy.retrievers.hybrid import (
    RaggyHybridRetriever,
    RaggyScoredNode,
    RaggyBM25Indexer,
)
from raggy.agents.reflective import (
    RaggyReflectiveAgent,
    RaggyResponse,
)
from raggy.server.api import (
    run_raggy_server,
    RaggyServiceState,
)

# Core abstractions (re-exported for seamless drop-in compatibility)
try:
    from llama_index.core import (
        VectorStoreIndex,
        SummaryIndex,
        DocumentSummaryIndex,
        KeywordTableIndex,
        SimpleDirectoryReader,
        Document,
        Settings,
        PromptTemplate,
        StorageContext,
        Response,
    )
except ImportError:
    # Fallback placeholders if optional dependencies are not yet installed
    VectorStoreIndex = None
    SummaryIndex = None
    DocumentSummaryIndex = None
    KeywordTableIndex = None
    SimpleDirectoryReader = None
    Document = None
    Settings = None
    PromptTemplate = None
    StorageContext = None
    Response = None

__all__ = [
    "__version__",
    "config",
    "RaggyConfig",
    "RaggyChunk",
    "RaggySmartChunker",
    "smart_chunk_document",
    "RaggyHybridRetriever",
    "RaggyScoredNode",
    "RaggyBM25Indexer",
    "RaggyReflectiveAgent",
    "RaggyResponse",
    "run_raggy_server",
    "RaggyServiceState",
    "VectorStoreIndex",
    "SummaryIndex",
    "DocumentSummaryIndex",
    "KeywordTableIndex",
    "SimpleDirectoryReader",
    "Document",
    "Settings",
    "PromptTemplate",
    "StorageContext",
    "Response",
]

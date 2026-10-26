#hello4: Raggy Top-Level Package Init
#hello4: The Next-Generation Modular RAG & Agent Framework

import sys
import os
from pathlib import Path

#hello4: Dynamically locate and resolve local raggy-core
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CORE_PATH = _REPO_ROOT / "raggy-core"
if _CORE_PATH.exists() and str(_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(_CORE_PATH))

#hello4: Raggy package version
__version__ = "1.0.0"

#hello4: Expose custom Raggy components
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

#hello4: Core abstractions re-exported for seamless drop-in compatibility
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
    #hello4: Fallback placeholders if dependencies are not yet installed
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

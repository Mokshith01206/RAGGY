"""
Raggy Configuration Module.
Provides centralized settings, model defaults, and custom tuning parameters for Raggy.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RaggyConfig:
    """Global configuration for Raggy Framework."""
    
    version: str = "1.0.0"
    app_name: str = "Raggy"
    
    # Model defaults
    default_llm_model: str = os.getenv("RAGGY_LLM_MODEL", "gpt-4o-mini")
    default_embed_model: str = os.getenv("RAGGY_EMBED_MODEL", "text-embedding-3-small")
    default_temperature: float = float(os.getenv("RAGGY_TEMPERATURE", "0.1"))
    
    # Chunker defaults
    chunk_size: int = int(os.getenv("RAGGY_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("RAGGY_CHUNK_OVERLAP", "50"))
    enable_smart_hierarchy: bool = True
    
    # Retrieval defaults
    similarity_top_k: int = int(os.getenv("RAGGY_TOP_K", "5"))
    hybrid_alpha: float = float(os.getenv("RAGGY_HYBRID_ALPHA", "0.6"))  # Weight for dense vs sparse
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    
    # Agent & Reflection defaults
    confidence_threshold: float = float(os.getenv("RAGGY_CONFIDENCE_THRESHOLD", "0.65"))
    max_reflection_retries: int = 2
    
    # Server defaults
    server_host: str = os.getenv("RAGGY_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("RAGGY_PORT", "8000"))
    
    # Persistence
    storage_dir: str = os.getenv("RAGGY_STORAGE_DIR", "./.raggy_storage")
    
    extra: Dict[str, Any] = field(default_factory=dict)


# Default global singleton
config = RaggyConfig()

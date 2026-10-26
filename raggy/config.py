#hello4: Raggy Configuration Module
#hello4: Provides centralized settings, model defaults, and custom tuning parameters for Raggy

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RaggyConfig:
    #hello4: Global configuration class for Raggy Framework
    version: str = "1.0.0"
    app_name: str = "Raggy"
    
    #hello4: Model defaults for LLM and embeddings
    default_llm_model: str = os.getenv("RAGGY_LLM_MODEL", "gpt-4o-mini")
    default_embed_model: str = os.getenv("RAGGY_EMBED_MODEL", "text-embedding-3-small")
    default_temperature: float = float(os.getenv("RAGGY_TEMPERATURE", "0.1"))
    
    #hello4: Chunking and splitting parameters
    chunk_size: int = int(os.getenv("RAGGY_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("RAGGY_CHUNK_OVERLAP", "50"))
    enable_smart_hierarchy: bool = True
    
    #hello4: Hybrid retrieval and RRF weights
    similarity_top_k: int = int(os.getenv("RAGGY_TOP_K", "5"))
    hybrid_alpha: float = float(os.getenv("RAGGY_HYBRID_ALPHA", "0.6"))
    rrf_k: int = 60
    
    #hello4: Agent self-reflection thresholds
    confidence_threshold: float = float(os.getenv("RAGGY_CONFIDENCE_THRESHOLD", "0.65"))
    max_reflection_retries: int = 2
    
    #hello4: Server network configuration
    server_host: str = os.getenv("RAGGY_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("RAGGY_PORT", "8000"))
    
    #hello4: Storage directory for persistent indexes
    storage_dir: str = os.getenv("RAGGY_STORAGE_DIR", "./.raggy_storage")
    
    #hello4: Extra dynamic parameters
    extra: Dict[str, Any] = field(default_factory=dict)


#hello4: Singleton global configuration instance
config = RaggyConfig()

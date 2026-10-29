#hello4: Raggy Document Extractor
#hello4: Multi-format document text extraction and parser for RAG pipelines

import os
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from raggy.chunking.smart_chunker import RaggySmartChunker, RaggyChunk


class RaggyDocumentExtractor:
    #hello4: Universal text extractor supporting PDF, TXT, MD, CSV, JSON, and source code

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        #hello4: Initialize chunker
        self.chunker = RaggySmartChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_breadcrumb_prefix=True,
        )

    def extract_from_pdf(self, file_source: Union[str, bytes]) -> Dict[str, Any]:
        #hello4: Extract text from PDF files or raw byte streams
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF extraction. Install via: pip install pypdf")

        if isinstance(file_source, bytes):
            reader = pypdf.PdfReader(io.BytesIO(file_source))
        else:
            reader = pypdf.PdfReader(file_source)

        page_texts = []
        for idx, page in enumerate(reader.pages, 1):
            extracted = page.extract_text() or ""
            if extracted.strip():
                page_texts.append(f"## Page {idx}\n{extracted.strip()}")

        full_text = "\n\n".join(page_texts)
        return {
            "text": full_text,
            "page_count": len(reader.pages),
            "total_chars": len(full_text),
            "format": "pdf",
        }

    def extract_from_text(self, text_content: str, filename: str = "document.txt") -> Dict[str, Any]:
        #hello4: Process raw text or markdown
        return {
            "text": text_content,
            "page_count": 1,
            "total_chars": len(text_content),
            "format": "text",
            "filename": filename,
        }

    def extract_file(self, file_path_or_name: str, file_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        #hello4: Auto-detect file format and extract structured content
        filename = Path(file_path_or_name).name
        ext = Path(file_path_or_name).suffix.lower()

        if ext == ".pdf":
            if file_bytes is not None:
                res = self.extract_from_pdf(file_bytes)
            else:
                res = self.extract_from_pdf(file_path_or_name)
        else:
            if file_bytes is not None:
                text = file_bytes.decode("utf-8", errors="ignore")
            else:
                with open(file_path_or_name, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            res = {
                "text": text,
                "page_count": 1,
                "total_chars": len(text),
                "format": ext.lstrip(".") or "txt",
            }

        res["filename"] = filename
        return res

    def extract_and_chunk(self, file_path_or_name: str, file_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        #hello4: Extract text and generate contextual chunks for RAG indexing
        extracted = self.extract_file(file_path_or_name, file_bytes=file_bytes)
        filename = extracted["filename"]
        text = extracted["text"]

        chunks = self.chunker.chunk_text(text, doc_id=filename)
        extracted["chunks"] = chunks
        extracted["chunk_count"] = len(chunks)
        return extracted


#hello4: Default global extractor instance
document_extractor = RaggyDocumentExtractor()

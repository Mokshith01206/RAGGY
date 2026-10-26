#hello4: Raggy Smart Chunker
#hello4: Hierarchical context-preserving text splitter for advanced document ingestion

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RaggyChunk:
    #hello4: Data container representing an enriched chunk with breadcrumbs and metadata
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None


class RaggySmartChunker:
    #hello4: Intelligent hierarchical chunker preserving headers and document structure

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        include_breadcrumb_prefix: bool = True,
    ):
        #hello4: Initialize chunker settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_breadcrumb_prefix = include_breadcrumb_prefix

    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        #hello4: Parse markdown headings and maintain breadcrumb trails
        lines = text.split("\n")
        sections = []
        current_title = "Root"
        current_level = 0
        current_lines: List[str] = []
        breadcrumbs = ["Root"]

        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = header_pattern.match(line.strip())
            if match:
                #hello4: Flush previous section text
                if current_lines:
                    sections.append({
                        "title": current_title,
                        "breadcrumb": " > ".join(breadcrumbs),
                        "text": "\n".join(current_lines).strip()
                    })
                    current_lines = []

                level = len(match.group(1))
                title = match.group(2).strip()

                #hello4: Adjust hierarchy levels
                while len(breadcrumbs) > level:
                    breadcrumbs.pop()
                if len(breadcrumbs) < level:
                    breadcrumbs.append(title)
                else:
                    breadcrumbs[-1] = title

                current_title = title
                current_level = level
            else:
                current_lines.append(line)

        #hello4: Append final trailing section
        if current_lines:
            sections.append({
                "title": current_title,
                "breadcrumb": " > ".join(breadcrumbs),
                "text": "\n".join(current_lines).strip()
            })

        return sections

    def _split_into_windows(self, text: str) -> List[str]:
        #hello4: Windowing strategy with overlap for long text blocks
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            sub_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(sub_words))
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def chunk_text(self, text: str, doc_id: str = "doc_0") -> List[RaggyChunk]:
        #hello4: Primary entrypoint to turn raw text into enriched RaggyChunks
        sections = self._extract_sections(text)
        result_chunks: List[RaggyChunk] = []

        chunk_counter = 0
        for sec_idx, sec in enumerate(sections):
            sec_text = sec["text"]
            if not sec_text:
                continue

            windows = self._split_into_windows(sec_text)
            for win_idx, window in enumerate(windows):
                chunk_counter += 1
                #hello4: Prepend contextual breadcrumb to chunk text
                prefix = f"[{sec['breadcrumb']}]\n" if self.include_breadcrumb_prefix else ""
                enriched_text = f"{prefix}{window}"

                chunk_metadata = {
                    "doc_id": doc_id,
                    "section_title": sec["title"],
                    "breadcrumb": sec["breadcrumb"],
                    "section_index": sec_idx,
                    "chunk_in_section": win_idx,
                    "chunk_id": f"{doc_id}_c{chunk_counter}",
                }

                result_chunks.append(
                    RaggyChunk(
                        text=enriched_text,
                        metadata=chunk_metadata,
                        chunk_id=chunk_metadata["chunk_id"],
                    )
                )

        return result_chunks


def smart_chunk_document(doc_text: str, doc_name: str = "document") -> List[RaggyChunk]:
    #hello4: Convenience wrapper function
    chunker = RaggySmartChunker()
    return chunker.chunk_text(doc_text, doc_id=doc_name)

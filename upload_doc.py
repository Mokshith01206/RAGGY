#hello4: Raggy Document Upload & Extraction Runner
#hello4: Ingests documents (PDF, TXT, MD, CSV, etc.), extracts text, indexes into RAG, and opens interactive Q&A

import sys
import os
from pathlib import Path

#hello4: Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raggy import (
    RaggyDocumentExtractor,
    RaggyHybridRetriever,
    RaggyReflectiveAgent,
    config,
)

def main():
    #hello4: Print welcome banner
    print("=" * 65)
    print("  [RAGGY] Document Upload, Text Extraction & RAG Q&A")
    print(f"  Framework Version: {config.version}")
    print("=" * 65)

    #hello4: Get document path from argv or prompt user
    if len(sys.argv) > 1:
        doc_path = sys.argv[1].strip().strip('"').strip("'")
    else:
        print("\nEnter or drag-and-drop the path to your document (PDF, TXT, MD, etc.):")
        doc_path = input("File Path > ").strip().strip('"').strip("'")

    if not doc_path:
        print("[!] No file path provided. Exiting.")
        return

    if not os.path.exists(doc_path):
        print(f"[!] File not found: {doc_path}")
        return

    #hello4: Step 1 - Extract Text and Generate Context Chunks
    print(f"\n[*] Extracting text from '{os.path.basename(doc_path)}'...")
    extractor = RaggyDocumentExtractor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    
    try:
        extracted = extractor.extract_and_chunk(doc_path)
    except Exception as e:
        print(f"[!] Extraction failed: {e}")
        return

    chunks = extracted["chunks"]
    print(f"[+] Extraction complete!")
    print(f"    - Format:      {extracted['format'].upper()}")
    print(f"    - Pages:       {extracted['page_count']}")
    print(f"    - Total Chars: {extracted['total_chars']}")
    print(f"    - Chunks Made: {len(chunks)}")

    #hello4: Step 2 - Display Text Preview
    print("\n" + "-" * 65)
    print("[Extracted Text Preview]:")
    preview = extracted["text"][:350].strip()
    print(preview + ("\n... [remaining text indexed into RAG]" if len(extracted["text"]) > 350 else ""))
    print("-" * 65)

    #hello4: Step 3 - Index Chunks into Hybrid Retriever
    print("\n[*] Indexing chunks into Raggy Hybrid Retriever (BM25 + RRF)...")
    corpus = [{"id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks]
    retriever = RaggyHybridRetriever(alpha=config.hybrid_alpha, similarity_top_k=3)
    retriever.set_corpus(corpus)

    #hello4: Step 4 - Initialize Reflective Agent
    agent = RaggyReflectiveAgent(
        retriever=retriever,
        confidence_threshold=config.confidence_threshold,
        max_iterations=config.max_reflection_retries,
    )
    print("[+] Ready for questions!\n")

    #hello4: Step 5 - Interactive Q&A Loop
    print("=" * 65)
    print(f"  Ask questions about: {os.path.basename(doc_path)}")
    print("  (Type 'exit' or 'q' to quit)")
    print("=" * 65)

    while True:
        try:
            query = input("\nAsk > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            response = agent.query(query)
            print(f"\n[Answer]:\n{response.response}")
            if response.source_nodes:
                print(f"\n[Source Context ({len(response.source_nodes)} matches, confidence: {response.confidence_score:.2f})]:")
                for i, node in enumerate(response.source_nodes[:2], 1):
                    breadcrumb = node.metadata.get("breadcrumb", "Doc")
                    print(f"  [{i}] ({breadcrumb}) {node.text[:100]}...")
            print("-" * 65)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    #hello4: Run main entrypoint
    main()

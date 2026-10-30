#hello4: Raggy End-to-End Live Execution Demo
#hello4: Tests hierarchical smart chunking, hybrid BM25 retrieval, and reflective agent reasoning

import sys
from pathlib import Path

#hello4: Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raggy import (
    RaggySmartChunker,
    RaggyHybridRetriever,
    RaggyReflectiveAgent,
    config,
)

def run_demo():
    #hello4: Print banner
    print("=" * 60)
    print("  [RAGGY] Live Execution & Verification Demo")
    print(f"  Framework Version: {config.version}")
    print("=" * 60)

    #hello4: Sample knowledge base text with markdown hierarchy
    sample_document = """
# Apollo Space Program
## Overview
The Apollo program was the third United States human spaceflight program carried out by NASA.
It succeeded in landing the first humans on the Moon in 1969.

## Key Missions
### Apollo 11
Apollo 11 was the spaceflight that first landed humans on the Moon on July 20, 1969.
Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew.

### Apollo 13
Apollo 13 was the seventh crewed mission in the Apollo space program.
An oxygen tank exploded two days after launch, but the crew returned safely to Earth.

## Technological Legacy
The Apollo program spurred advances in many areas of technology, including avionics,
telecommunications, computers, and fuel cell energy storage.
"""

    #hello4: Step 1 - Smart Hierarchical Chunking
    print("\n[Step 1] Running RaggySmartChunker...")
    chunker = RaggySmartChunker(chunk_size=120, chunk_overlap=20, include_breadcrumb_prefix=True)
    chunks = chunker.chunk_text(sample_document, doc_id="apollo_doc")
    print(f"[+] Successfully generated {len(chunks)} contextual chunks:")
    for idx, c in enumerate(chunks, 1):
        breadcrumb = c.metadata.get("breadcrumb", "")
        print(f"    - Chunk {idx} [{breadcrumb}]: {c.text[:65].strip()}...")

    #hello4: Step 2 - Build Corpus and Initialize Hybrid Retriever
    print("\n[Step 2] Building corpus and configuring RaggyHybridRetriever (RRF)...")
    corpus = [{"id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks]
    retriever = RaggyHybridRetriever(alpha=0.6, similarity_top_k=3)
    retriever.set_corpus(corpus)
    print(f"[+] BM25 index built with {len(corpus)} documents.")

    #hello4: Step 3 - Initialize Reflective Agent
    print("\n[Step 3] Initializing RaggyReflectiveAgent...")
    agent = RaggyReflectiveAgent(
        retriever=retriever,
        confidence_threshold=0.6,
        max_iterations=2
    )

    #hello4: Step 4 - Execute Test Queries
    test_queries = [
        "Who landed on the Moon in Apollo 11?",
        "What happened during Apollo 13?",
        "What technologies were developed by Apollo?",
    ]

    for q in test_queries:
        print("\n" + "-" * 60)
        print(f"[*] Query: '{q}'")
        res = agent.query(q)
        print(f"    - Retrieval Confidence: {res.confidence_score:.2f}")
        print(f"    - Reflection Passes:    {res.reflection_count}")
        print(f"    - Matches Found:        {len(res.source_nodes)}")
        if res.source_nodes:
            top_hit = res.source_nodes[0]
            print(f"    - Top Match Score:      {top_hit.score:.4f}")
            print(f"    - Top Match Path:       {top_hit.metadata.get('breadcrumb')}")
        print(f"\n[Raggy Output]:\n{res.response}")

        print("  SUCCESS All Raggy components executed successfully!")

if __name__ == "__main__":
    #hello4: Run main execution
    run_demo()

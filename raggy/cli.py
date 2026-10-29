#hello4: Raggy CLI Module
#hello4: Unified command line interface for the Raggy RAG Framework

import sys
import os
import argparse
from raggy.config import config
from raggy.retrievers.hybrid import RaggyHybridRetriever
from raggy.agents.reflective import RaggyReflectiveAgent
from raggy.chunking.smart_chunker import RaggySmartChunker
from raggy.server.api import run_raggy_server


def cmd_info(args):
    #hello4: Display active environment, model configurations, and thresholds
    print("=" * 55)
    print("  [RAGGY] Next-Gen Modular RAG & Agent Framework")
    print("=" * 55)
    print(f"  Version:               {config.version}")
    print(f"  Default LLM:           {config.default_llm_model}")
    print(f"  Default Embeddings:    {config.default_embed_model}")
    print(f"  Chunk Size / Overlap:  {config.chunk_size} / {config.chunk_overlap}")
    print(f"  Hybrid Alpha (Dense):  {config.hybrid_alpha}")
    print(f"  Confidence Threshold:  {config.confidence_threshold}")
    print(f"  Storage Directory:     {config.storage_dir}")
    print("=" * 55)


def cmd_serve(args):
    #hello4: Launch the lightweight REST API server
    port = args.port or config.server_port
    host = args.host or config.server_host
    run_raggy_server(host=host, port=port)


def cmd_chat(args):
    #hello4: Launch interactive terminal REPL conversation session
    print("=" * 55)
    print("  [RAGGY] Interactive Chat Session")
    print("  Type 'exit' or 'quit' to end session.")
    print("=" * 55)

    retriever = RaggyHybridRetriever(alpha=config.hybrid_alpha)
    chunker = RaggySmartChunker(chunk_size=config.chunk_size)
    agent = RaggyReflectiveAgent(retriever)
    corpus = []

    #hello4: Index input directory documents if provided
    if args.docs and os.path.isdir(args.docs):
        print(f"[*] Indexing documents from '{args.docs}'...")
        count = 0
        for root, _, files in os.walk(args.docs):
            for file in files:
                if file.endswith((".txt", ".md", ".py", ".json")):
                    fpath = os.path.join(root, file)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        chunks = chunker.chunk_text(text, doc_id=file)
                        for c in chunks:
                            corpus.append({"id": c.chunk_id, "text": c.text, "metadata": c.metadata})
                            count += 1
                    except Exception as e:
                        print(f"  [!] Could not read {file}: {e}")
        retriever.set_corpus(corpus)
        print(f"[+] Ingested {count} chunks across {len(corpus)} segments.\n")

    #hello4: Interactive chat loop
    while True:
        try:
            query = input("Raggy > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            response = agent.query(query)
            print(f"\n[Answer]\n{response.response}\n")
            if response.source_nodes:
                print(f"[Sources] ({len(response.source_nodes)} matches, confidence: {response.confidence_score:.2f}):")
                for i, n in enumerate(response.source_nodes[:3], 1):
                    breadcrumb = n.metadata.get("breadcrumb", "Unknown")
                    print(f"  [{i}] ({breadcrumb}) {n.text[:80]}...")
            print("-" * 55)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def cmd_upload(args):
    #hello4: Extract and chat with a single uploaded document
    from raggy.readers.extractor import RaggyDocumentExtractor
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    print(f"[*] Extracting text from '{os.path.basename(file_path)}'...")
    extractor = RaggyDocumentExtractor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    extracted = extractor.extract_and_chunk(file_path)
    chunks = extracted["chunks"]
    print(f"[+] Extracted {extracted['total_chars']} chars across {len(chunks)} chunks (Format: {extracted['format'].upper()}, Pages: {extracted['page_count']})")

    retriever = RaggyHybridRetriever(alpha=config.hybrid_alpha)
    retriever.set_corpus([{"id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks])
    agent = RaggyReflectiveAgent(retriever)

    print("\n" + "=" * 55)
    print(f"  [RAGGY] Q&A over '{os.path.basename(file_path)}'")
    print("  Type 'exit' or 'q' to end.")
    print("=" * 55)

    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q or q.lower() in ["exit", "quit", "q"]:
                break
            res = agent.query(q)
            print(f"\n[Answer]:\n{res.response}")
            if res.source_nodes:
                print(f"[Sources]: {res.source_nodes[0].metadata.get('breadcrumb', 'Doc')}")
            print("-" * 55)
        except (KeyboardInterrupt, EOFError):
            break


def cmd_query(args):
    #hello4: One-shot query evaluation
    retriever = RaggyHybridRetriever()
    agent = RaggyReflectiveAgent(retriever)
    res = agent.query(args.query)
    print(res.response)


def main():
    #hello4: Main CLI parser entrypoint
    parser = argparse.ArgumentParser(
        prog="raggy",
        description="Raggy: Next-Generation Modular RAG Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    #hello4: Register subcommand parsers
    subparsers.add_parser("info", help="Display Raggy environment and configuration info")

    serve_parser = subparsers.add_parser("serve", help="Run the Raggy REST API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument("--docs", help="Path to document folder to index for chat")

    query_parser = subparsers.add_parser("query", help="Execute a one-shot query")
    query_parser.add_argument("query", help="Query string")

    upload_parser = subparsers.add_parser("upload", help="Upload a document, extract text, and start RAG Q&A")
    upload_parser.add_argument("file", help="Path to PDF, TXT, MD, CSV, or JSON document")

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        cmd_info(args)
        parser.print_help()


if __name__ == "__main__":
    main()

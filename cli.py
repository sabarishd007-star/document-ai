"""
DocuMind CLI - Command-Line Interface for Document RAG with Gemini
"""
import sys
import os
import argparse

# Ensure proper encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser(description="DocuMind CLI - Document RAG with Gemini")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: run (start server)
    parser_run = subparsers.add_parser("run", help="Start the FastAPI backend and web UI")
    parser_run.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser_run.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser_run.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")

    # Command: status
    subparsers.add_parser("status", help="Check vector index status")

    # Command: upload
    parser_upload = subparsers.add_parser("upload", help="Upload and index a .pdf or .txt file")
    parser_upload.add_argument("file_path", help="Path to the document to index")

    # Command: query
    parser_query = subparsers.add_parser("query", help="Ask a question about indexed documents")
    parser_query.add_argument("question", help="Question string")
    parser_query.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")

    # Command: reset
    subparsers.add_parser("reset", help="Reset all indexed vectors and metadata")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        import uvicorn
        print(f"[*] Starting DocuMind on http://{args.host}:{args.port}")
        print(f"[*] Web UI available at: http://{args.host}:{args.port}/ui")
        print(f"[*] API documentation at: http://{args.host}:{args.port}/docs")
        uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command == "status":
        from backend.app.services.vector_store import VectorStore
        from backend.app.config import settings
        store = VectorStore()
        store.load()
        count = store.index.ntotal if store.index is not None else 0
        print(f"[+] Total Chunks Indexed:  {count}")
        print(f"[+] Embedding Model:      {settings.embedding_model}")
        print(f"[+] Similarity Threshold: {settings.similarity_threshold}")
        print(f"[+] Storage Path:         {settings.vector_store_path}")

    elif args.command == "upload":
        if not os.path.exists(args.file_path):
            print(f"[-] Error: File '{args.file_path}' does not exist.")
            sys.exit(1)

        from backend.app.services.ingestion import extract_and_chunk_document
        from backend.app.services.vector_store import VectorStore

        filename = os.path.basename(args.file_path)
        with open(args.file_path, "rb") as f:
            raw_bytes = f.read()

        store = VectorStore()
        store.load()

        if store.is_duplicate_file(raw_bytes):
            print(f"[!] File '{filename}' is already indexed. Reset first if you want to re-upload.")
            sys.exit(0)

        print(f"[*] Extracting and chunking '{filename}' with sentence boundaries & page tracking...")
        chunks = extract_and_chunk_document(raw_bytes, filename, chunk_size=1000, overlap=200)
        print(f"[*] Split into {len(chunks)} chunks.")

        store.register_file_hash(raw_bytes)
        added = store.add_chunks(chunks)
        store.save()
        print(f"[+] Successfully indexed {added} chunks into vector store! (Total: {store.index.ntotal})")

    elif args.command == "query":
        from backend.app.services.vector_store import VectorStore
        from backend.app.services.rag import RAGService

        store = VectorStore()
        if not store.load() or store.index.ntotal == 0:
            print("[!] Vector store is empty. Please upload documents first with: python cli.py upload <file>")
            sys.exit(0)

        rag = RAGService(vector_store=store)
        print(f"[*] Searching context and querying Gemini for: \"{args.question}\"...\n")
        res = rag.answer_query(query=args.question, top_k=args.top_k)

        print("=" * 60)
        print("Answer:")
        print(res["answer"])
        print("=" * 60)
        if res.get("citations"):
            print("\nCitations:")
            for c in res["citations"]:
                print(f" - [{c['chunk_id']}] {c['filename']} (similarity: {c['similarity_score'] * 100:.1f}%)")

    elif args.command == "reset":
        from backend.app.services.vector_store import VectorStore
        store = VectorStore()
        store.clear()
        store.save()
        print("[+] Vector store index reset successfully.")

if __name__ == "__main__":
    main()

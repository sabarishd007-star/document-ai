import os
import logging
import hashlib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Configure logging so RAG stage timings appear in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

from backend.app.config import settings
from backend.app.utils.errors import register_exception_handlers, DocuMindException
from backend.app.services.ingestion import validate_and_extract_text
from backend.app.services.chunking import chunk_text
from backend.app.services.vector_store import VectorStore
from backend.app.services.rag import RAGService

import threading
from contextlib import asynccontextmanager

# Shared singletons
vector_store = VectorStore()
# Attempt loading existing index if present on disk
vector_store.load()
rag_service = RAGService(vector_store=vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm embedding model in a background thread so server boots instantly."""
    def _warmup():
        try:
            logging.info("Pre-warming embedding model in background thread...")
            _ = vector_store.model
            logging.info("Embedding model pre-warmed successfully (dimension: %s).", vector_store.dimension)
        except Exception as e:
            logging.warning("Embedding model pre-warming deferred: %s", e)

    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title="DocuMind API",
    description="Intelligent Document RAG Assistant with Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# Register exception handlers
register_exception_handlers(app)

# Enable CORS for local dev / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask about the documents")
    top_k: Optional[int] = Field(10, ge=1, le=25, description="Number of context chunks to retrieve")


@app.get("/")
def root():
    return {
        "app": "DocuMind API",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/status")
def get_index_status():
    total_vectors = vector_store.index.ntotal if vector_store.index is not None else 0
    # Collect unique indexed files from metadata
    file_counts = {}
    for meta in vector_store.metadata_store:
        fn = meta.get("filename", "unknown")
        file_counts[fn] = file_counts.get(fn, 0) + 1
    files_list = [{"filename": fn, "name": fn, "chunks": count} for fn, count in file_counts.items()]

    return {
        "total_chunks_indexed": total_vectors,
        "chunks_indexed": total_vectors,
        "chunks": total_vectors,
        "total_chunks": total_vectors,
        "embedding_model": settings.embedding_model,
        "model": settings.gemini_model,
        "vector_store_path": settings.vector_store_path,
        "similarity_threshold": settings.similarity_threshold,
        "min_similarity": settings.similarity_threshold,
        "files": files_list,
        "documents": files_list
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a .pdf or .txt document, extract text, chunk it, and index it into the vector store.
    Returns 409 if this exact file (by SHA-256) has already been indexed.
    """
    filename = file.filename or "unknown"

    # 1. Read raw bytes first so we can hash before heavy processing
    raw_bytes = await file.read()

    # 2. Duplicate-file guard (SHA-256 of raw content)
    if vector_store.is_duplicate_file(raw_bytes):
        raise DocuMindException(
            f"'{filename}' is already indexed. Reset the index first if you want to re-upload it.",
            status_code=409
        )

    # 3. Extract and chunk with page-number tracking
    from backend.app.services.ingestion import extract_and_chunk_document
    chunks = extract_and_chunk_document(raw_bytes, filename, chunk_size=1000, overlap=200)

    if not chunks:
        raise DocuMindException("Document resulted in 0 chunks after text extraction.", status_code=400)

    # 5. Register hash, add chunks, persist
    vector_store.register_file_hash(raw_bytes)
    added = vector_store.add_chunks(chunks)
    vector_store.save()

    return {
        "message": f"Successfully processed and indexed '{filename}'.",
        "filename": filename,
        "chunks_indexed": added,
        "chunks": added,
        "chunks_created": added,
        "total_index_size": vector_store.index.ntotal
    }


@app.post("/api/query")
def query_documents(request: QueryRequest):
    """
    Ask a question against all indexed documents (non-streaming).
    """
    result = rag_service.answer_query(query=request.query, top_k=request.top_k)
    return result


@app.post("/api/query/stream")
def query_documents_stream(request: QueryRequest):
    """
    Ask a question against indexed documents with real-time token streaming via SSE.
    """
    return StreamingResponse(
        rag_service.answer_query_stream(query=request.query, top_k=request.top_k),
        media_type="text/event-stream"
    )


@app.delete("/api/index")
def clear_index():
    """
    Clears all documents and vectors from the vector store.
    """
    vector_store.clear()
    vector_store.save()
    return {"message": "Vector store index has been reset successfully."}


# Convenience alias routes matching frontend defaults
@app.post("/upload")
async def upload_alias(file: UploadFile = File(...)):
    return await upload_document(file=file)


@app.post("/query")
def query_alias(request: QueryRequest):
    return query_documents(request=request)


@app.post("/query/stream")
def query_stream_alias(request: QueryRequest):
    return query_documents_stream(request=request)


@app.delete("/index")
def index_alias():
    return clear_index()

@app.get("/overview")
def overview_redirect():
    return RedirectResponse(url="/ui/#overview")


@app.get("/chat")
def chat_redirect():
    return RedirectResponse(url="/ui/#chat")


# Mount frontend web interface
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_frontend_dir = os.path.join(_base_dir, "frontend")
if os.path.exists(_frontend_dir):
    app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="frontend")



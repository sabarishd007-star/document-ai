import os
import re
import time
import logging
import threading
from typing import Dict, Any, List
from google import genai
from backend.app.config import settings
from backend.app.utils.errors import DocuMindException

logger = logging.getLogger("documind.rag")

RAG_SYSTEM_PROMPT = """You are an expert AI technical study assistant. Answer the user question using ONLY the provided document chunks below.

Answer Quality Rules:
1. Strict Grounding: Answer using ONLY the retrieved document context. Do not invent equations, examples, facts, or assumptions outside the text.
2. Complete Coverage: Include all relevant information available in the retrieved chunks. When asked about a model, concept, or architecture, always include its definition, components, working mechanism, advantages, limitations, and applications whenever the document provides them.
3. Detail Preservation: Do not omit important technical points, benefits, equations, or lists just to make the answer shorter. Include all specific types (e.g. Sparse, Denoising, Contractive, VAE) and parameters (e.g. L1 sparsity, L2 small weights, Jacobian Frobenius penalty) described in the chunks.
4. Precise Attribution: Keep models and concepts strictly separated. Do not mix or transfer properties, limitations, or advantages of one model to another (e.g., Boltzmann Machine vs. Deep Boltzmann Machine vs. Deep Belief Network).
5. Exact Terminology: Preserve the exact terminology, definitions, parameter names, and organization used in the source document.
6. Structured Presentation: Present answers with clear markdown headings, bullet points, and numbered lists for readability.
7. Missing Information: If the retrieved context does not contain certain details (e.g., a specific formula or limitation), explicitly state what information is missing instead of guessing or filling in from general knowledge.
8. Clean Output: Do NOT mention or print source file names, document titles, or page numbers (such as '(filename.pdf, Page X)', '[dev unit-2.pdf]', or '[1]') anywhere in your answer text.
"""

LLM_TIMEOUT_SECONDS = 45
LLM_MAX_RETRIES = 1
SNIPPET_LENGTH = 140  # characters shown in the citation card


def clean_citations(text: str) -> str:
    """Removes inline file citations like (file.pdf, Page 5) or [1] from text."""
    if not text:
        return ""
    text = re.sub(r'\s*\*\(?(?:Source|Cites?):?[^\n\r]*?\)\*?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\[[^\]\r\n]*?(?:\.pdf|Page\s*\d+)[^\]\r\n]*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\([^\)\r\n]*?(?:\.pdf|Page\s*\d+)[^\)\r\n]*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\[\d+(?:\s*,\s*\d+)*\](?:\s*[^,\n\r]*?\.pdf,\s*Page\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\[\d+\]', '', text)
    return text.strip()


class RAGService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._client = None
        self._last_key = None

    def _get_api_key(self) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
        if not api_key or api_key == "your_gemini_api_key_here":
            try:
                from dotenv import dotenv_values
                vals = dotenv_values(".env")
                api_key = vals.get("GEMINI_API_KEY", "")
            except Exception:
                pass
        return api_key

    @property
    def client(self) -> genai.Client:
        api_key = self._get_api_key()
        if not api_key or api_key == "your_gemini_api_key_here":
            raise DocuMindException(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file.",
                status_code=500
            )
        if self._client is None or self._last_key != api_key:
            self._client = genai.Client(api_key=api_key)
            self._last_key = api_key
        return self._client

    def _call_gemini_with_timeout(self, model: str, prompt: str) -> str:
        """Calls Gemini with a hard timeout and one retry on failure/timeout."""
        result: Dict[str, Any] = {"text": None, "error": None}

        def _run():
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                result["text"] = response.text if response.text else ""
            except Exception as e:
                result["error"] = str(e)

        for attempt in range(LLM_MAX_RETRIES + 1):
            t0 = time.perf_counter()
            result = {"text": None, "error": None}
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            thread.join(timeout=LLM_TIMEOUT_SECONDS)
            elapsed = time.perf_counter() - t0

            if thread.is_alive():
                logger.warning("Gemini timed out after %.1fs (attempt %d)", elapsed, attempt + 1)
                if attempt < LLM_MAX_RETRIES:
                    continue
                raise DocuMindException(
                    f"Gemini did not respond within {LLM_TIMEOUT_SECONDS}s. Please try again shortly.",
                    status_code=504
                )

            if result["error"]:
                logger.error("Gemini error (attempt %d): %s", attempt + 1, result["error"])
                if attempt < LLM_MAX_RETRIES:
                    continue
                raise DocuMindException(
                    f"Error querying Gemini model: {result['error']}",
                    status_code=502
                )

            logger.info("Gemini responded in %.2fs (attempt %d)", elapsed, attempt + 1)
            return result["text"]

    def answer_query(self, query: str, top_k: int = None, model_name: str = None) -> Dict[str, Any]:
        """
        Retrieves unique context chunks and generates a grounded, numbered-citation answer.
        Each stage is timed and logged.
        """
        target_model = model_name or getattr(settings, "gemini_model", "gemini-3.5-flash")
        k = top_k or getattr(settings, "retrieval_top_k", 6)

        if not query.strip():
            raise DocuMindException("Query string cannot be empty.", status_code=400)

        # Stage 1: Vector search with built-in deduplication
        t0 = time.perf_counter()
        chunks = self.vector_store.search(query, top_k=k)
        search_ms = (time.perf_counter() - t0) * 1000
        logger.info("FAISS search: %.1f ms, %d unique chunks retrieved", search_ms, len(chunks))

        if not chunks:
            return {
                "answer": "No relevant content found in the uploaded documents for this query.",
                "citations": [],
                "context_used": [],
                "timings_ms": {"search": round(search_ms, 1), "llm": 0}
            }

        # Stage 2: Build numbered grounded prompt with page references
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("filename", "unknown")
            page_num = chunk.get("page_number")
            page_info = f", Page {page_num}" if page_num else ""
            context_blocks.append(f"[{i}] ({source}{page_info}):\n{chunk['text']}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            f"Context Chunks:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Stage 3: LLM generation with timeout + retry
        t1 = time.perf_counter()
        answer_text = self._call_gemini_with_timeout(target_model, prompt)
        llm_ms = (time.perf_counter() - t1) * 1000
        logger.info("LLM generation: %.1f ms with model %s", llm_ms, target_model)

        if not answer_text:
            answer_text = "Unable to generate a response."
        else:
            answer_text = clean_citations(answer_text)

        # Build citation cards with snippet preview and page numbers
        citations = [
            {
                "number": i,
                "chunk_id": chunk.get("chunk_id"),
                "filename": chunk.get("filename"),
                "page": chunk.get("page_number"),
                "page_number": chunk.get("page_number"),
                "score": round(chunk.get("similarity_score", 0.0), 4),
                "similarity_score": round(chunk.get("similarity_score", 0.0), 4),
                "text": chunk["text"],
                "snippet": chunk["text"][:SNIPPET_LENGTH].rstrip() + (
                    "…" if len(chunk["text"]) > SNIPPET_LENGTH else ""
                )
            }
            for i, chunk in enumerate(chunks, 1)
        ]

        return {
            "answer": answer_text,
            "citations": citations,
            "sources": citations,
            "context_used": [c["text"] for c in chunks],
            "timings_ms": {"search": round(search_ms, 1), "llm": round(llm_ms, 1)}
        }

    def answer_query_stream(self, query: str, top_k: int = 5):
        """
        Retrieves relevant context chunks and streams Gemini tokens in real-time.
        Yields Server-Sent Event (SSE) formatted strings:
          data: {"type": "sources", "citations": [...]}\n\n
          data: {"type": "token", "token": "..."}\n\n
          data: {"type": "done", "latency_ms": ...}\n\n
        """
        import json
        t0 = time.perf_counter()
        target_model = settings.gemini_model or "gemini-3.5-flash"

        # Stage 1: Retrieval
        chunks = self.vector_store.search(query=query, top_k=top_k)
        search_ms = (time.perf_counter() - t0) * 1000
        logger.info("Streaming search retrieved %d chunks in %.1f ms", len(chunks), search_ms)

        citations = [
            {
                "number": i,
                "chunk_id": chunk.get("chunk_id"),
                "filename": chunk.get("filename"),
                "page": chunk.get("page_number"),
                "page_number": chunk.get("page_number"),
                "score": round(chunk.get("similarity_score", 0.0), 4),
                "similarity_score": round(chunk.get("similarity_score", 0.0), 4),
                "text": chunk["text"],
                "snippet": chunk["text"][:SNIPPET_LENGTH].rstrip() + (
                    "…" if len(chunk["text"]) > SNIPPET_LENGTH else ""
                )
            }
            for i, chunk in enumerate(chunks, 1)
        ]

        # Send citations first
        yield f"data: {json.dumps({'type': 'sources', 'citations': citations})}\n\n"

        if not chunks:
            yield f"data: {json.dumps({'type': 'token', 'token': 'No relevant document content found to answer your question.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'latency_ms': round((time.perf_counter() - t0) * 1000, 1)})}\n\n"
            return

        # Stage 2: Prompt assembly
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f", Page {chunk['page_number']}" if chunk.get("page_number") else ""
            context_blocks.append(f"Source [{i}] ({chunk['filename']}{page_info}):\n{chunk['text']}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            f"Context Chunks:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Stage 3: Streaming generation
        try:
            stream_response = self.client.models.generate_content_stream(
                model=target_model,
                contents=prompt,
            )
            for chunk in stream_response:
                if chunk.text:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk.text})}\n\n"
        except Exception as e:
            logger.error("Streaming error from Gemini: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        total_ms = (time.perf_counter() - t0) * 1000
        yield f"data: {json.dumps({'type': 'done', 'latency_ms': round(total_ms, 1)})}\n\n"

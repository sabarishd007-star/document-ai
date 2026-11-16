import os
import io
from typing import List, Dict, Any
from fastapi import UploadFile
from pypdf import PdfReader
from backend.app.utils.errors import DocuMindException
from backend.app.services.chunking import chunk_text

MAX_FILE_SIZE_MB = 10
MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def extract_pages_from_bytes(contents: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Validates file and extracts text organized by page number.
    Returns: List of dicts with {"page_number": int, "text": str}
    """
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    if ext not in ["txt", "pdf"]:
        raise DocuMindException(f"Invalid file type '.{ext}'. Only .txt and .pdf are allowed.", status_code=400)

    if len(contents) == 0:
        raise DocuMindException(f"The file '{filename}' is empty.", status_code=400)
    
    if len(contents) > MAX_BYTES:
        raise DocuMindException(f"File '{filename}' exceeds the maximum allowed size of {MAX_FILE_SIZE_MB}MB.", status_code=413)

    pages = []
    if ext == "txt":
        try:
            raw_text = contents.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = contents.decode("latin-1")
            except Exception:
                raise DocuMindException(f"Could not decode text file '{filename}'.", status_code=400)
        
        if not raw_text.strip():
            raise DocuMindException(f"No extractable text found in '{filename}'.", status_code=400)
            
        pages.append({"page_number": 1, "text": raw_text.strip()})

    elif ext == "pdf":
        try:
            pdf_file = io.BytesIO(contents)
            reader = PdfReader(pdf_file)
            if len(reader.pages) == 0:
                raise DocuMindException(f"PDF file '{filename}' has no pages.", status_code=400)

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append({"page_number": page_num, "text": page_text.strip()})

        except Exception as e:
            if isinstance(e, DocuMindException):
                raise
            raise DocuMindException(f"Error reading PDF file '{filename}': {str(e)}", status_code=400)

        if not pages:
            raise DocuMindException(f"No extractable text found in '{filename}'. It might be a scanned PDF without OCR.", status_code=400)

    return pages

def extract_text_from_bytes(contents: bytes, filename: str) -> str:
    """
    Validates file type, size, and decodes content from bytes for .txt and .pdf files.
    """
    pages = extract_pages_from_bytes(contents, filename)
    return "\n\n".join(p["text"] for p in pages)

def extract_and_chunk_document(
    contents: bytes,
    filename: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Extracts pages and chunks each page preserving exact page numbers in metadata.
    """
    pages = extract_pages_from_bytes(contents, filename)
    doc_id = filename.replace(" ", "_")
    all_chunks = []
    global_index = 0

    for p in pages:
        page_num = p["page_number"]
        page_text = p["text"]
        
        page_chunks = chunk_text(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            metadata={
                "filename": filename,
                "doc_id": doc_id,
                "page_number": page_num
            }
        )
        
        for c in page_chunks:
            c["chunk_index"] = global_index
            global_index += 1
            all_chunks.append(c)

    return all_chunks

async def validate_and_extract_text(file: UploadFile) -> str:
    filename = file.filename or ""
    contents = await file.read()
    return extract_text_from_bytes(contents, filename)

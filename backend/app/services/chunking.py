import re
from typing import List, Dict, Any, Optional
from backend.app.utils.errors import DocuMindException

def chunk_text_by_sentences(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits text into chunks respecting sentence and paragraph boundaries.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # Split text into sentences using standard punctuation boundaries
    sentence_endings = re.compile(r'(?<=[.!?\n\n])\s+')
    raw_sentences = sentence_endings.split(clean_text)
    
    # Filter out empty splits
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return []

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If a single sentence is larger than chunk_size, split it into character slices
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Chunk the oversized sentence
            sub_start = 0
            while sub_start < len(sentence):
                sub_end = min(sub_start + chunk_size, len(sentence))
                chunks.append(sentence[sub_start:sub_end])
                if sub_end >= len(sentence):
                    break
                sub_start += max(1, chunk_size - overlap)
            continue

        if len(current_chunk) + len(sentence) + (1 if current_chunk else 0) <= chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # Handle overlap by keeping the tail of the previous chunk
            if overlap > 0 and len(current_chunk) >= overlap:
                overlap_text = current_chunk[-overlap:].strip()
            else:
                overlap_text = ""
                
            current_chunk = (overlap_text + " " if overlap_text else "") + sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Split input text into overlapping chunks using sentence-aware boundary snapping.
    
    Args:
        text: The raw text string to chunk.
        chunk_size: Target maximum length for each chunk (default: 1000).
        chunk_overlap: Target overlap length between consecutive chunks (default: 200).
        metadata: Optional dictionary with document metadata (filename, page_number, etc.)
        
    Returns:
        List of dictionaries with 'chunk_id', 'text', and associated metadata.
    """
    if chunk_size <= 0:
        raise DocuMindException("chunk_size must be a positive integer.", status_code=400)
    
    if chunk_overlap < 0:
        raise DocuMindException("chunk_overlap must be non-negative.", status_code=400)
        
    if chunk_overlap >= chunk_size:
        raise DocuMindException("chunk_overlap must be strictly less than chunk_size.", status_code=400)

    clean_text = text.strip()
    if not clean_text:
        return []

    str_chunks = chunk_text_by_sentences(clean_text, chunk_size=chunk_size, overlap=chunk_overlap)
    doc_meta = metadata or {}
    page_suffix = f"_p{doc_meta['page_number']}" if "page_number" in doc_meta else ""
    doc_id = doc_meta.get("doc_id", "doc")

    chunks = []
    char_cursor = 0
    for idx, content in enumerate(str_chunks):
        start_char = clean_text.find(content[:min(30, len(content))], char_cursor)
        if start_char == -1:
            start_char = char_cursor
        end_char = start_char + len(content)
        char_cursor = max(char_cursor, start_char)

        chunk_data = {
            "chunk_id": f"{doc_id}{page_suffix}_chunk_{idx}",
            "chunk_index": idx,
            "text": content,
            "start_char": start_char,
            "end_char": end_char,
            **doc_meta
        }
        chunks.append(chunk_data)

    return chunks

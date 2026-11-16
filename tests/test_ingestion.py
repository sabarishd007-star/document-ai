import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.app.services.ingestion import extract_text_from_bytes, validate_and_extract_text, MAX_BYTES
from backend.app.utils.errors import DocuMindException

def test_invalid_file_extension():
    with pytest.raises(DocuMindException) as exc:
        extract_text_from_bytes(b"content", "file.docx")
    assert exc.value.status_code == 400
    assert "Invalid file type '.docx'" in exc.value.message

def test_empty_file():
    with pytest.raises(DocuMindException) as exc:
        extract_text_from_bytes(b"", "sample.txt")
    assert exc.value.status_code == 400
    assert "is empty" in exc.value.message

def test_file_exceeds_max_size():
    huge_data = b"x" * (MAX_BYTES + 10)
    with pytest.raises(DocuMindException) as exc:
        extract_text_from_bytes(huge_data, "large.txt")
    assert exc.value.status_code == 413
    assert "exceeds the maximum allowed size" in exc.value.message

def test_valid_txt_file_utf8():
    content = b"Welcome to DocuMind text extraction testing."
    extracted = extract_text_from_bytes(content, "sample.txt")
    assert extracted == "Welcome to DocuMind text extraction testing."

def test_valid_txt_file_latin1():
    # Character 0xe9 is 'é' in latin-1, invalid in strict utf-8
    latin1_bytes = b"DocuMind r\xe9sum\xe9 test"
    extracted = extract_text_from_bytes(latin1_bytes, "resume.txt")
    assert "DocuMind" in extracted

def test_whitespace_only_file():
    with pytest.raises(DocuMindException) as exc:
        extract_text_from_bytes(b"   \n\t  ", "blank.txt")
    assert exc.value.status_code == 400
    assert "No extractable text found" in exc.value.message



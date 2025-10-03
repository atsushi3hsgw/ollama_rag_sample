import pdfplumber
import os
import re

def load_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf_text(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def split_text(text: str, max_length: int = 300) -> list:
    """
    Split text into chunks of approximately max_length characters.
    """
    chunks = []
    current = ""
    for line in text.splitlines():
        if len(current) + len(line) < max_length:
            current += line + " "
        else:
            chunks.append(current.strip())
            current = line + " "
    if current:
        chunks.append(current.strip())
    return chunks

def split_text_with_window(text: str, window_size: int = 3, stride: int = 2) -> list:
    """
    Divide into sentences and chunk using a sliding window.
    Supports Japanese punctuation marks (.!?).
    """

    sentences = re.split(r'(?<=[。！？\.\?!])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(0, len(sentences) - window_size + 1, stride):
        chunk = " ".join(sentences[i:i+window_size])
        chunks.append(chunk)

    if len(sentences) % stride != 0:
        chunk = " ".join(sentences[-window_size:])
        if chunk not in chunks:
            chunks.append(chunk)

    return chunks

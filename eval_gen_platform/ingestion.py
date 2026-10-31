import hashlib
import os
import subprocess
from urllib.parse import unquote, urlparse

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    WikipediaLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import cache
from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_FILE_MB, UPLOAD_DIR
from loader import transcribe_file
from logger import get_logger

log = get_logger("ingestion")

CODE_EXT = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs", ".php", ".html", ".css", ".sql", ".sh"}
TEXT_EXT = {".txt", ".md"}
SUPPORTED_EXT = {".pdf", ".docx", ".pptx", ".ppt", ".wav"} | TEXT_EXT | CODE_EXT


# ------------------------- loaders (one per format) -------------------------
def load_pptx(path: str) -> list[Document]:
    from pptx import Presentation

    docs = []
    for number, slide in enumerate(Presentation(path).slides, start=1):
        texts = [s.text for s in slide.shapes if s.has_text_frame and s.text.strip()]
        if texts:
            docs.append(Document(page_content="\n".join(texts), metadata={"slide": number}))
    return docs


def convert_ppt_to_pptx(path: str) -> str:
    """Old .ppt files are converted with LibreOffice (sudo apt install libreoffice-impress)."""
    try:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pptx", "--outdir", os.path.dirname(path), path],
            check=True, capture_output=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        raise ValueError("Could not convert .ppt. Install LibreOffice or save the file as .pptx.")
    return os.path.splitext(path)[0] + ".pptx"


def load_wav(path: str) -> list[Document]:
    """Speech-to-text (loader.py, same as the voice project) so the audio becomes searchable text."""
    with open(path, "rb") as f:
        text, _language = transcribe_file(f.read(), os.path.basename(path))
    return [Document(page_content=text)]


def load_file(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext == ".docx":
        return Docx2txtLoader(path).load()
    if ext == ".ppt":
        return load_pptx(convert_ppt_to_pptx(path))
    if ext == ".pptx":
        return load_pptx(path)
    if ext == ".wav":
        return load_wav(path)
    return TextLoader(path, autodetect_encoding=True).load()  # txt, md and code files


def _process(kb, doc_id: str, name: str, docs: list[Document]) -> str:
    if cache.is_ingested(doc_id):
        log.info(f"Document cache HIT: {name}")
        return f"'{name}' is already in the knowledge base (skipped, served from Redis cache)."

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError(f"No readable text found in '{name}'.")

    for d in docs:
        d.metadata["source"] = name

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    kb.add(chunks)
    cache.mark_ingested(doc_id, name, len(chunks))
    log.info(f"Ingested '{name}' ({len(chunks)} chunks)")
    return f"Added '{name}' ({len(chunks)} chunks)."


def ingest_file(kb, filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type '{ext}'.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"'{filename}' is larger than {MAX_FILE_MB} MB.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, os.path.basename(filename))
    with open(path, "wb") as f:
        f.write(data)

    doc_id = hashlib.sha256(data).hexdigest()
    if cache.is_ingested(doc_id):  # skip the heavy work completely
        return _process(kb, doc_id, filename, [])
    return _process(kb, doc_id, filename, load_file(path))


def ingest_url(kb, url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Please enter a valid URL starting with http:// or https://")
    doc_id = hashlib.sha256(f"url:{url}".encode()).hexdigest()
    if cache.is_ingested(doc_id):
        return _process(kb, doc_id, url, [])
    return _process(kb, doc_id, url, WebBaseLoader(url).load())


def ingest_wikipedia(kb, page: str) -> str:
    """Accepts a page title ('Alan Turing') or a full Wikipedia URL."""
    title = page.strip()
    if "wikipedia.org" in title:
        title = unquote(title.rstrip("/").split("/")[-1]).replace("_", " ")
    if not title:
        raise ValueError("Please enter a Wikipedia page title or URL.")
    doc_id = hashlib.sha256(f"wiki:{title.lower()}".encode()).hexdigest()
    if cache.is_ingested(doc_id):
        return _process(kb, doc_id, f"Wikipedia: {title}", [])
    docs = WikipediaLoader(query=title, load_max_docs=1).load()
    return _process(kb, doc_id, f"Wikipedia: {title}", docs)

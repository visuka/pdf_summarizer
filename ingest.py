import os
import json
import hashlib
import logging
import re
from typing import Generator, List, Dict, Any

import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ===== CONFIG =====
PDF_DIR = "./pdfs"
PERSIST_DIR = "./chroma_db"
STATE_FILE = "./ingested.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 500  # Chroma performs better with moderate batches


# ===== SETUP LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ===== UTILS =====
def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean_text(text: str) -> str:
    """Clean raw PDF text: fix hyphenation, normalize whitespace, strip."""
    # Fix hyphenated line breaks: "learn-\ning" → "learning"
    text = re.sub(r'-\s*\n\s*', '', text)
    # Replace newline + space combos with single space
    text = re.sub(r'\n\s*', ' ', text)
    # Collapse all whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def pdf_pages(pdf_path: str) -> Generator[tuple[int, str], None, None]:
    """Yield (page_number, cleaned_text) for each page."""
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            raw_text = page.get_text()
            clean = clean_text(raw_text)
            yield i + 1, clean
        doc.close()
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise


def load_state() -> Dict[str, str]:
    """Load state dict; handle missing/corrupted JSON safely."""
    if not os.path.exists(STATE_FILE):
        logger.info(f"No state file found. Starting fresh.")
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                logger.warning(f"State file {STATE_FILE} is empty. Treating as new.")
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load state from {STATE_FILE}: {e}. Starting fresh.")
        return {}


def save_state(state: Dict[str, str]) -> None:
    """Save state with UTF-8 and proper formatting."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logger.debug(f"State saved to {STATE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        raise


def batched(iterable, n: int):
    """Batch iterable into chunks of size n."""
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:i + n]


# ===== MAIN =====
def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    state = load_state()

    # Auto-detect device: CUDA > MPS > CPU
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass  # torch not installed — stay on CPU

    logger.info(f"Using embedding device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}  # improves similarity
    )

    # Load or create Chroma DB
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDFs found in {PDF_DIR}")
        return

    total_added = 0

    for fname in sorted(pdf_files):
        path = os.path.join(PDF_DIR, fname)

        # Skip if file disappeared between listing and processing
        if not os.path.isfile(path):
            logger.warning(f"File vanished: {fname}")
            continue

        try:
            file_hash = sha256_file(path)
        except Exception as e:
            logger.error(f"Failed to hash {fname}: {e}")
            continue

        # Skip if already ingested with same hash
        if state.get(fname) == file_hash:
            logger.info(f"SKIP (already ingested): {fname}")
            continue

        logger.info(f"INGEST: {fname}")

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        try:
            for page_no, page_text in pdf_pages(path):
                if not page_text:
                    continue

                chunks = splitter.split_text(page_text)
                for ci, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    texts.append(chunk)
                    metadatas.append({
                        "source": fname,
                        "sha256": file_hash,
                        "page": page_no
                    })
                    ids.append(f"{fname}:{file_hash}:p{page_no}:c{ci}")

        except Exception as e:
            logger.error(f"Skipping {fname} due to extraction error: {e}")
            continue

        if not texts:
            logger.warning(f"No text extracted from {fname}")
            continue

        # Add in batches to avoid memory issues
        try:
            for batch_texts, batch_metas, batch_ids in zip(
                batched(texts, BATCH_SIZE),
                batched(metadatas, BATCH_SIZE),
                batched(ids, BATCH_SIZE)
            ):
                vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            vectorstore.persist()
            added = len(texts)
            total_added += added
            logger.info(f"  Added {added} chunks")

            # ✅ Only mark as ingested *after* successful DB update
            state[fname] = file_hash
            save_state(state)

        except Exception as e:
            logger.error(f"Failed to ingest {fname} into vector DB: {e}")
            continue

    logger.info(f"DONE. Total new chunks added: {total_added}")


if __name__ == "__main__":
    main()
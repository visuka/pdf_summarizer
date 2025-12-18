# retriever.py
import os
import logging
import argparse
from typing import List, Dict, Any, Optional  # ✅ Fixed: use Dict, List from typing

# Disable Chroma telemetry (optional but recommended)
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from langchain_chroma import Chroma  # ✅ Modern import (after `pip install langchain-chroma`)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ===== CONFIG =====
PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ===== EXPORTED FUNCTIONS =====
__all__ = ["retrieve_context", "format_for_llm"]


def retrieve_context(
    query: str,
    k: int = 4,
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve and structure context for LLM prompting.
    
    Args:
        query: Search query string
        k: Max number of chunks to return (after filtering)
        score_threshold: Min cosine similarity (0.0–1.0). Higher = more relevant.
        metadata_filter: Chroma-style filter, e.g., {"source": "notes.pdf"}
    
    Returns:
        List of dicts with keys: content, source, page, score
    """
    # Auto-detect device: CUDA > MPS > CPU
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass

    logger.info(f"Loading embeddings on device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}  # enables cosine similarity [0,1]
    )

    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    except Exception as e:
        logger.error(f"Failed to load Chroma DB: {e}")
        return []

    # Fetch more candidates if we'll filter by score
    fetch_k = k * 2 if score_threshold is not None else k
    fetch_k = max(fetch_k, k)  # ensure at least k

    try:
        # ✅ Use similarity_search_with_score for actual relevance scores
        results_with_scores = vectorstore.similarity_search_with_score(
            query=query,
            k=fetch_k,
            filter=metadata_filter
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

    # ✅ Manually filter by score (Chroma doesn't support score_threshold in query)
    filtered_results = []
    for doc, score in results_with_scores:
        # With normalize_embeddings=True → cosine similarity (0 to 1)
        # Higher score = more relevant
        if score_threshold is not None and score < score_threshold:
            continue
        if len(filtered_results) >= k:
            break

        filtered_results.append({
            "content": doc.page_content.strip(),
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": round(float(score), 4)
        })

    logger.info(
        f"Retrieved {len(filtered_results)} chunks (k={k}, threshold={score_threshold}) "
        f"for query: '{query[:50]}...'"
    )
    return filtered_results


def format_for_llm(context: List[Dict[str, Any]], max_chunks: Optional[int] = None) -> str:
    """
    Format retrieved context into a clean, citation-style string for LLM prompts.
    
    Example:
    [1] (notes.pdf, p.5): Bagging trains models on bootstrap samples...
    [2] (ml.pdf, p.12): It reduces variance by averaging...
    """
    if not context:
        return "No relevant context found."

    if max_chunks and len(context) > max_chunks:
        context = context[:max_chunks]

    lines = []
    for i, item in enumerate(context, 1):
        src = item["source"]
        pg = item["page"]
        content = item["content"]
        lines.append(f"[{i}] ({src}, p.{pg}): {content}")

    return "\n\n".join(lines)


# ===== CLI FOR TESTING =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve context from Chroma DB")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("-k", "--top_k", type=int, default=4, help="Number of results")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="Min cosine similarity (e.g., 0.3). Higher = more relevant.")
    parser.add_argument("-s", "--source", type=str, default=None,
                        help="Filter by PDF filename (exact match)")

    args = parser.parse_args()

    metadata_filter = {"source": args.source} if args.source else None

    context = retrieve_context(
        query=args.query,
        k=args.top_k,
        score_threshold=args.threshold,
        metadata_filter=metadata_filter
    )

    print("\n🔍 Retrieved Context:\n")
    print(format_for_llm(context))
    
    print("\n📊 Metadata Summary:")
    for i, c in enumerate(context, 1):
        print(f"[{i}] {c['source']} p.{c['page']} | score: {c['score']}")
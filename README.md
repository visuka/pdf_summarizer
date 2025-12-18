# Local PDF RAG System (Chroma + Sentence Transformers + Qwen)

This project builds a fully local Retrieval-Augmented Generation (RAG) system that answers questions using information from PDF documents.

The system reads PDFs, cleans and splits their text, converts the text into semantic embeddings, stores them in a vector database, retrieves the most relevant chunks for a question, and finally sends that context to a local LLM to generate an answer.

No cloud APIs are used. Everything runs locally.

---

## How the system works (end-to-end)

PDFs
↓
Text cleaning and chunking
↓
Sentence Transformer embeddings
↓
Chroma vector database (persistent)
↓
Similarity search
↓
Relevant text context
↓
Qwen LLM via Ollama
↓
Final answer

---

## Code layout and responsibility

ingest.py  
Reads PDF files, cleans the text, splits it into chunks, creates embeddings, and stores them in ChromaDB.  
Tracks already-ingested PDFs using a hash file so the same document is never embedded twice.

retriever.py  
Loads the vector database, embeds the user query, performs similarity search, filters results by score, and formats the retrieved chunks for LLM prompting.

query.py  
Orchestrates the RAG flow: retrieves relevant context, builds a grounded prompt, and sends it to a locally running Qwen model via Ollama.

Supporting folders and files:

pdfs/ → place all PDF documents here  
chroma_db/ → persistent vector database  
ingested.json → tracks which PDFs are already embedded  

---

## How ingestion works (ingest.py)

1. PDFs are read using PyMuPDF.
2. Raw text is cleaned (hyphen fixes, whitespace normalization).
3. Text is split into overlapping chunks (500 characters with 50 overlap).
4. Each chunk is converted into a deterministic embedding using sentence-transformers/all-MiniLM-L6-v2.
5. Chunks, embeddings, and metadata (PDF name, page number, hash) are stored in ChromaDB.
6. Each PDF’s SHA-256 hash is saved so it is not re-processed.

Run ingestion:

python ingest.py

You can add new PDFs later and re-run the script; only new or modified files will be embedded.

---

## How retrieval works (retriever.py)

1. The user query is embedded using the same sentence-transformer model.
2. ChromaDB performs a cosine-similarity search.
3. The top-K most similar chunks are retrieved.
4. Results can be filtered by similarity score or metadata.
5. Retrieved chunks are formatted into a clean context for the LLM.

Example:

python retriever.py "Explain scaling in pods" -k 5 -t 0.3

---

## How question answering works (query.py)

1. A user question is defined.
2. Relevant context is retrieved from ChromaDB.
3. A strict RAG prompt is built.
4. The prompt is sent to Qwen 1.8B via Ollama.
5. The response is streamed to the console.

Run:

python query.py

---

## Design principles

Encoders retrieve, decoders generate.  
Geometry (vectors) before language.  
Retrieval quality > LLM size.  
Local, inspectable, debuggable system.

---

## Example RAG prompt

You are a helpful AI assistant.
Answer the question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't know based on the provided context."

Context:
[1] (Kubernetes.pdf, p.12): A pod is the smallest deployable unit...

Question:
Explain scaling in pods

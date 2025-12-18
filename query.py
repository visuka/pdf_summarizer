# query.py
import requests
import json
import sys
from typing import Optional

# Import your retriever
from retriever import retrieve_context, format_for_llm


# ===== CONFIG =====
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen:1.8b"
TIMEOUT = 120  # seconds


def query_qwen(prompt: str, stream: bool = True) -> Optional[str]:
    """
    Send prompt to locally running Qwen 1.8B via Ollama.
    
    Args:
        prompt: Full prompt (context + question)
        stream: Whether to stream output to console (True) or return full response (False)
    
    Returns:
        Full response string if stream=False, else None (output printed)
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.3,      # Lower = more deterministic (good for RAG)
            "top_p": 0.9,
            "num_ctx": 2048,         # Qwen 1.8B supports up to 32K, but 2K is safe
        }
    }

    try:
        if stream:
            print("\nLLM is thinking...\n")
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                full_response = []
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            text = chunk["response"]
                            print(text, end="", flush=True)
                            full_response.append(text)
                        if chunk.get("done"):
                            break
                print("\n")  # final newline
                return "".join(full_response)
        else:
            # Non-streaming (for programmatic use)
            r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            response = r.json()
            return response.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Is it running?", file=sys.stderr)
        print("   → Run: `ollama serve` or restart Ollama app", file=sys.stderr)
        return None
    except requests.exceptions.Timeout:
        print("Error: Ollama request timed out. Model may be loading or system is slow.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error calling Qwen: {e}", file=sys.stderr)
        return None


def main():
    # Your query
    query = "Explain scaling in pods"

    # Step 1: Retrieve context
    print(f"🔍 Retrieving context for: '{query}'")
    context = retrieve_context(query, k=3, score_threshold=0.25)
    
    if not context:
        print("⚠️ No relevant context found. Proceeding with LLM-only answer.")
        context_str = "No context available."
    else:
        context_str = format_for_llm(context)

    # Step 2: Build RAG prompt
    prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't know based on the provided context."

Context:
{context_str}

Question: {query}

Answer:"""

    print("\n" + "="*60)
    print("FINAL PROMPT SENT TO QWEN:")
    print("="*60)
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    print("="*60 + "\n")

    # Step 3: Query Qwen
    response = query_qwen(prompt, stream=True)

    if response is None:
        sys.exit(1)

    # Optional: Save to file
    # with open("answer.txt", "w", encoding="utf-8") as f:
    #     f.write(f"Q: {query}\n\nA: {response}\n")


if __name__ == "__main__":
    main()
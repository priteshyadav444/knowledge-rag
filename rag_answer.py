"""
Tiny RAG, stage 3: retrieve relevant chunks, then let an LLM write the answer.

Needs:
  - rag_step1.py in the same folder (we reuse its document loading)
  - Ollama running locally with a model pulled, e.g.:  ollama pull llama3.2
"""
import json
import urllib.request

import numpy as np
from sentence_transformers import SentenceTransformer

# Reuse the settings and chunk loader we already wrote in stage 1.
from rag_step1 import MODEL_NAME, TOP_K, load_chunks

OLLAMA_URL = "http://localhost:11434/api/chat"  # Ollama's local address
LLM_MODEL = "qwen3:8b"                         # the LLM that writes answers
MIN_SCORE = 0.3  # below this, treat a chunk as "not relevant enough"


def retrieve(question, model, chunks, doc_vectors):
    """The 'R' in RAG: find the most relevant chunks (same as stage 1)."""
    q_vector = model.encode(question, normalize_embeddings=True)
    scores = doc_vectors @ q_vector
    best = np.argsort(scores)[::-1][:TOP_K]
    return [(chunks[i], float(scores[i])) for i in best if scores[i] >= MIN_SCORE]


def build_prompt(question, results):
    """The 'A' in RAG: augment the question with the retrieved documentation."""
    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk, _ in results
    )
    return (
        "Answer the question using ONLY the documentation below.\n"
        "If the documentation does not contain the answer, say "
        "\"I couldn't find that in the documentation.\" Do not make things up.\n\n"
        f"DOCUMENTATION:\n{context}\n\n"
        f"QUESTION: {question}"
    )


def ask_llm(prompt):
    """The 'G' in RAG: send the prompt to the local LLM and return its answer."""
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,  # wait for the full answer instead of word-by-word
        "think": False,   # Qwen3: skip the "thinking out loud" part, just answer
    }).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["message"]["content"]


def main():
    print(f"Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks()
    doc_vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    print(f"Ready: {len(chunks)} chunks indexed.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        results = retrieve(question, model, chunks, doc_vectors)
        if not results:
            print("\nNo relevant documentation found.\n")
            continue

        print("\nThinking... (can take a while on CPU)")
        try:
            answer = ask_llm(build_prompt(question, results))
        except Exception as error:
            print(f"Could not reach Ollama: {error}")
            print(f"Is it running? Try: ollama run {LLM_MODEL}\n")
            continue

        print(f"\nANSWER:\n{answer}\n")
        print("SOURCES USED:")
        for chunk, score in results:
            print(f"  - {chunk['source']} (score {score:.3f})")
        print()


if __name__ == "__main__":
    main()
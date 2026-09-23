"""
Tiny RAG, stage 3: retrieve relevant chunks, then let an LLM write the answer.

Needs:
  - rag_step1.py in the same folder (we reuse its document loading)
  - A GGUF model file locally (e.g., download from Hugging Face)
  - llama-cpp-python: pip install llama-cpp-python
"""
import numpy as np
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Reuse the settings and chunk loader we already wrote in stage 1.
from rag_step1 import MODEL_NAME, TOP_K, load_chunks

# Path to your local GGUF model file.
# Download one from: https://huggingface.co/search?inference=true&model_type=text-generation&quantization=gguf
MODEL_PATH = "./models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"  # adjust this path to your model
MIN_SCORE = 0.3  # below this, treat a chunk as "not relevant enough"


def missing_model_shards(model_path):
    """Return missing files when model_path points at a split GGUF."""
    path = Path(model_path)
    match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name)
    if not match:
        return []

    prefix, _, shard_count = match.groups()
    return [
        path.with_name(f"{prefix}-{number:05d}-of-{shard_count}.gguf")
        for number in range(1, int(shard_count) + 1)
        if not path.with_name(
            f"{prefix}-{number:05d}-of-{shard_count}.gguf"
        ).is_file()
    ]


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
        f"Documentation:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Combine all documentation passages that are relevant to the question "
        "into a clear, concise answer. Include distinct consequences and do not "
        "merely copy the first matching passage."
    )


def ask_llm(prompt, llm):
    """The 'G' in RAG: send the prompt to the local LLM and return its answer."""
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied documentation. "
                    "If it does not contain the answer, reply exactly: "
                    "I couldn't find that in the documentation. "
                    "Do not repeat the question, headings, or source labels. "
                    "Synthesize all relevant facts; do not stop after the first one."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        temperature=0.1,
        top_p=0.95,
        repeat_penalty=1.1,
    )
    return response["choices"][0]["message"]["content"].strip()


def main():
    print(f"Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks()
    doc_vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    print(f"Ready: {len(chunks)} chunks indexed.\n")

    print(f"Loading LLM from {MODEL_PATH}...")
    missing_shards = missing_model_shards(MODEL_PATH)
    if missing_shards:
        print("ERROR: This is a split GGUF model, and these shards are missing:")
        for shard in missing_shards:
            print(f"  - {shard}")
        print("Download all shards into the same directory, then run this script again.")
        return

    try:
        llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, verbose=False)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Download a GGUF model from: https://huggingface.co/search?inference=true&model_type=text-generation&quantization=gguf")
        print("Then update MODEL_PATH in this script.")
        return
    
    print("Ready!\n")

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
            answer = ask_llm(build_prompt(question, results), llm)
        except Exception as error:
            print(f"Error during LLM inference: {error}\n")
            continue

        print(f"\nANSWER:\n{answer}\n")
        print("SOURCES USED:")
        for chunk, score in results:
            print(f"  - {chunk['source']} (score {score:.3f})")
        print()


if __name__ == "__main__":
    main()

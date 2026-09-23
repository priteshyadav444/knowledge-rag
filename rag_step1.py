"""
Tiny RAG, stage 1: semantic search over your own documentation.

Put .md or .txt files in a folder called ./docs next to this script
(for example, pages exported from Archbee). If the folder is empty,
a few sample sentences are used instead.
"""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs"
MODEL_NAME = "all-MiniLM-L6-v2"  # small (~90 MB), runs fine on CPU
TOP_K = 3                        # how many results to show

SAMPLE_DOCS = [
    "When an invoice is completely paid, its status becomes Paid.",
    "Fully paid invoices trigger the payment-completed workflow, which emails a receipt to the customer.",
    "A quote can be converted into an invoice once the customer accepts it.",
    "Partially paid invoices keep the status Partially Paid until the remaining balance is settled.",
    "Users can upload a profile picture from the Account Settings page.",
    "Admins can deactivate a user; deactivated users cannot log in but their data is kept.",
    "Overdue invoices are those past their due date with an outstanding balance.",
]


def load_chunks():
    """Read files from ./docs and split them into paragraph-sized chunks."""
    chunks = []
    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.rglob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            # Simple chunking: one chunk per paragraph (split on blank lines).
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 30:  # skip tiny fragments like lone headings
                    chunks.append({"source": path.name, "text": para})
    if not chunks:
        print("No files found in ./docs, using sample sentences.\n")
        chunks = [{"source": "sample", "text": t} for t in SAMPLE_DOCS]
    return chunks


def main():
    print(f"Loading model {MODEL_NAME} (first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    # normalize_embeddings=True makes a simple dot product equal cosine similarity
    doc_vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Done. Each chunk is now a list of {doc_vectors.shape[1]} numbers.\n")

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        q_vector = model.encode(question, normalize_embeddings=True)
        scores = doc_vectors @ q_vector            # similarity with every chunk
        best = np.argsort(scores)[::-1][:TOP_K]    # highest scores first

        for rank, i in enumerate(best, 1):
            print(f"\n{rank}. score={scores[i]:.3f}  [{chunks[i]['source']}]")
            print(f"   {chunks[i]['text'][:400]}")
        print()


if __name__ == "__main__":
    main()

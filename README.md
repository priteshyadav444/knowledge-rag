# Tiny RAG Lab

This project searches the Markdown and text files in `docs/` and can use a local Ollama model to answer questions from the retrieved content.

## Requirements

- Python 3.9 or newer
- `pip`
- [Ollama](https://ollama.com/) for generated answers (`rag_answer.py` only)

## Setup

Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy sentence-transformers
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The embedding model is downloaded automatically the first time a script runs.

## Option 1: Run semantic search

This retrieves the most relevant passages without using an LLM:

```bash
python rag_step1.py
```

Enter a question at the prompt. Enter `quit` or `exit` to stop.

## Option 2: Run RAG with generated answers

Install Ollama, then download the model used by this project:

```bash
ollama pull qwen3:8b
```

Make sure Ollama is running. In another terminal, activate the virtual environment and start the application:

```bash
source .venv/bin/activate
python rag_answer.py
```

Enter a question at the prompt. The application retrieves relevant documentation, asks the local model to answer from that context, and displays the sources it used.

## Use your own documents

Add `.md` or `.txt` files anywhere inside `docs/`, then restart the script. Each document is split into paragraphs and indexed when the program starts.

## Configuration

The main settings are near the top of the Python files:

- `MODEL_NAME` in `rag_step1.py`: embedding model
- `TOP_K` in `rag_step1.py`: number of retrieved passages
- `LLM_MODEL` in `rag_answer.py`: Ollama model
- `MIN_SCORE` in `rag_answer.py`: minimum relevance score

If you change `LLM_MODEL`, pull that model with Ollama before running the application.

## Common problems

- **`ModuleNotFoundError`**: activate `.venv` and rerun the `pip install` command.
- **Cannot reach Ollama**: start Ollama or run `ollama serve`, then try again.
- **Model not found**: run `ollama pull qwen3:8b`.
- **No relevant documentation found**: ask a more specific question or lower `MIN_SCORE` slightly.

# knowledge-rag

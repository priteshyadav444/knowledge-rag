# Tiny RAG Lab

This project searches the Markdown and text files in `docs/` and can use a local LLM to answer questions from the retrieved content.

## Requirements

- Python 3.9 or newer
- `pip`
- A GGUF model file (for generated answers with `rag_answer.py`)

## Setup

Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy sentence-transformers llama-cpp-python
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

First, download a GGUF model from [Hugging Face](https://huggingface.co/search?inference=true&model_type=text-generation&quantization=gguf):

**Recommended models:**
- **Qwen2.5-7B**: Fast, good quality → `TheBloke/Qwen2.5-7B-Instruct-GGUF`
- **Mistral-7B**: Solid general purpose → `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`  
- **Llama-2-7B**: Popular, well-tested → `TheBloke/Llama-2-7B-Chat-GGUF`

**Steps:**
1. Create a `models/` folder in this directory
2. Download a GGUF model file into `models/`
3. Update `MODEL_PATH` in `rag_answer.py` to match your model filename
4. Run:

```bash
source .venv/bin/activate
python3 rag_answer.py
```

Enter a question at the prompt. The application retrieves relevant documentation, asks the local model to answer from that context, and displays the sources it used.

## Use your own documents

Add `.md` or `.txt` files anywhere inside `docs/`, then restart the script. Each document is split into paragraphs and indexed when the program starts.

## Configuration

The main settings are near the top of the Python files:

- `MODEL_NAME` in `rag_step1.py`: embedding model
- `TOP_K` in `rag_step1.py`: number of retrieved passages
- `MODEL_PATH` in `rag_answer.py`: path to your GGUF model
- `MIN_SCORE` in `rag_answer.py`: minimum relevance score

## Common problems

- **`ModuleNotFoundError`**: activate `.venv` and rerun the `pip install` command.
- **`FileNotFoundError` for model**: make sure the model file path in `MODEL_PATH` is correct.
- **Slow inference**: try a smaller model (e.g., 7B instead of 13B) or enable GPU with `n_gpu_layers=-1` (requires CUDA).
- **No relevant documentation found**: ask a more specific question or lower `MIN_SCORE` slightly.

# knowledge-rag

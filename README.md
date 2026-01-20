# llm-runtime-local

A collection of scripts and examples to run local LLM-based pipelines and Retrieval-Augmented Generation (RAG) experiments on local resources. This repository contains utilities for downloading models, creating embeddings, building local retrieval stores, and running simple RAG demos for text, SQL, images, audio and video.

TL;DR
- Try the quickstart to run a local RAG demo.
- Explore the example scripts to adapt pipelines for your models and storage backend.

## Contents / Key files
- [localrag.py](https://github.com/Dhanush-sai-reddy/llm-runtime-local/blob/main/localrag.py) — Minimal local RAG demo / coordinator script.
- [llmforsql.py](https://github.com/Dhanush-sai-reddy/llm-runtime-local/blob/main/llmforsql.py) — Example integration for LLMs and SQL.
- [sql-and-rag/](https://github.com/Dhanush-sai-reddy/llm-runtime-local/tree/main/sql-and-rag) — SQL + RAG examples and helpers.
- [videorag.py](https://github.com/Dhanush-sai-reddy/llm-runtime-local/blob/main/videorag.py) — Example pipeline for video → embeddings → RAG.
- [qwenvisionlanguagemodel.py](https://github.com/Dhanush-sai-reddy/llm-runtime-local/blob/main/qwenvisionlanguagemodel.py) — Vision + language example for Qwen-like models.
- [hfdownloader/](https://github.com/Dhanush-sai-reddy/llm-runtime-local/tree/main/hfdownloader) — Utilities for downloading models from Hugging Face with your hftoken
- [milvusdb/](https://github.com/Dhanush-sai-reddy/llm-runtime-local/tree/main/milvusdb) — Example / helpers for Milvus vector DB integration.
- [vision rag/](https://github.com/Dhanush-sai-reddy/llm-runtime-local/tree/main/vision%20rag) — Image/vision RAG examples.
- [docs&imagestovoiceast.py](https://github.com/Dhanush-sai-reddy/llm-runtime-local/blob/main/docs%26imagestovoiceast.py) -Image and document rag with voice output and reranker model
- [qwen3multimediaembeddings.ipynb](https://github.com/Dhanush-sai-reddy/llm-runtime-local/blob/main/qwen3multimediaembeddings.ipynb) — Notebook for multimedia embeddings (one embeddings for text,image,video unlike a different pipeline for all).


> Note: The repository currently contains multiple example scripts. Read the top of each script to learn required dependencies and configurable options
> its advisable for running scripts in colab 

## Requirements
- Python 3.9+
- Typical Python dependencies (install per-script or project requirements). Common packages used in this ecosystem:
  - torch
  - transformers
  - sentence-transformers or other embedding libs
  - faiss-cpu or a vector DB client (Milvus client if using Milvus)
  - numpy, pandas, torchvision (for vision examples)
  - langchain
  - langraph
- GPU recommended for larger models


## Quickstart (example workflow)
1. Clone the repo
   ```
   git clone https://github.com/Dhanush-sai-reddy/llm-runtime-local.git
   cd llm-runtime-local
   ```
2. Install dependencies (example)
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install torch transformers sentence-transformers faiss-cpu numpy
   ```
3. Download a model or weights
   - Use the scripts under `hfdownloader/` or your preferred method to fetch model weights.
   - and specify model paths locally/in colab
4. Prepare data & embeddings
   - Run a script or notebook (e.g., `qwen3multimediaembeddings.ipynb`) to generate embeddings and store them in a vector index (FAISS, Milvus, etc).
5. Run a local RAG demo
   ```
   python localrag.py
   ```
   - Check the top of the script for available flags (model path, index path, etc).

Notes:
- For Milvus usage, see the `milvusdb/` helper files and ensure the Milvus server is running before connecting.

## Examples of how scripts fit together
- hfdownloader/ → download model weights (uses docker)
- Embedding scripts / notebooks → create dense vectors for documents or multimedia
- Vector DB (FAISS / Milvus) → store and index embeddings
- localrag.py / videorag.py → query embeddings, fetch context, and run the local LLM to synthesize answers

## Contributing
Contributions welcome. Suggested workflow:
1. Fork the repo
2. Create a branch: `feat/readme-improvements` 
3. Make changes and submit a PR with a clear description and examples

## Troubleshooting
- Model download errors: check authentication for private Hugging Face models or large file timeouts.
- OOM on large models: use smaller weights or enable CPU offload or quantization methods (bitsandbytes/quantization).
- Vector DB connection problems: confirm the DB server is running and the client versions are compatible.

## Contact
- Repo owner: Dhanush-sai-reddy — open an issue for questions or feature requests.


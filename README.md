# LLM Runtime Local

A collection of Python scripts and notebooks for executing Large Language Models (LLMs) locally. This repository supports tasks such as Retrieval-Augmented Generation (RAG) on documents and videos, SQL database interaction, and Vision-Language modeling.

## Features

- **Local RAG**: Chat with PDF documents using local LLMs via `localrag.py`.
- **SQL Interaction**: Generate SQL queries and interact with databases using `llmforsql.py`.
- **Video RAG**: Perform RAG tasks on video content using `videorag.py`.
- **Vision Language Models**:
  - `qwenvisionlanguagemodel.py`: Run Qwen-based vision-language tasks.
  - `qwen3multimediaembeddings.ipynb`: Notebook for multimedia embeddings.
- **Voice Assistant**: `docs&imagestovoiceast.py` for document/image-to-voice capabilities.

## Prerequisites

- **Python 3.10+**
- **Ollama**
- **Milvus Vector Database**

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Dhanush-sai-reddy/llm-runtime-local.git](https://github.com/Dhanush-sai-reddy/llm-runtime-local.git)
   cd llm-runtime-local
Create and activate a virtual environment:

Bash

python -m venv venv
source venv/bin/activate
Install dependencies:

Bash

pip install langchain langchain-community pymupdf chromadb pymilvus torch transformers sentence-transformers
Usage
Document RAG
Run the local RAG script to query PDF documents.

Bash

python localrag.py
SQL Generation
Run the SQL interaction script.

Bash

python llmforsql.py
Vision & Video
For vision-related tasks or video RAG:

Bash

python qwenvisionlanguagemodel.py
python videorag.py
Folder Structure
hfdownloader/: Scripts for downloading models from Hugging Face.

milvusdb/: Configurations for Milvus vector database.

llmforsql/ & sql-and-rag/: Specialized logic for SQL and RAG integration.

vision rag/: Resources for vision-based RAG.

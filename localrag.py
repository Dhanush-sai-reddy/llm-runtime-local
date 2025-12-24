import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- NEW IMPORTS FOR PDF ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
LLM_MODEL = "gemma3:1b"
EMBED_MODEL = "mxbai-embed-large"
PDF_PATH = r"C:\Users\dhanu\Downloads\FLAP-SAM.pdf" # This works



# Check if file exists before running
if not os.path.exists(PDF_PATH):
    print(f"ERROR: The file  was not found.")
    exit()

print(f"1. Connecting to models: {LLM_MODEL} and {EMBED_MODEL}...")
llm = ChatOllama(model=LLM_MODEL)
embed_model = OllamaEmbeddings(model=EMBED_MODEL)

# --- LOADING & SPLITTING ---
print(f"2. Loading PDF: {PDF_PATH}...")
loader = PyPDFLoader(PDF_PATH)
raw_docs = loader.load()

# 2. Gemma 3 has a memory limit; we only want to feed it the relevant parts.
print("   Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Size of each piece (characters)
    chunk_overlap=200,    # Overlap to keep context between chunks
    add_start_index=True
)
chunks = text_splitter.split_documents(raw_docs)
print(f"   Created {len(chunks)} chunks from the PDF.")

# --- INDEXING ---
print("3. Indexing data into Vector Store...")
# We use a new collection name so we don't mix with old tests
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    collection_name="pdf-rag-storage" 
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

# --- RAG CHAIN ---
template = """
You are a helpful assistant. Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't see that in the document."

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# --- ASK QUESTION ---
query = "What is the communication cost reduction achieved in this and how is that achieved?"

print(f"\n4. Asking Question: '{query}'\n")
print("--- Answer ---")
for chunk in rag_chain.stream(query):
    print(chunk.content, end="", flush=True)

print("\n\n--- Done ---")
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

LLM_MODEL="gemma3:1b"
EMBED_MODEL="mxbai-embed-large"
PDF_PATH=r"C:\Users\dhanu\Downloads\FLAP-SAM.pdf" 

if not os.path.exists(PDF_PATH):
    print(f"ERROR: File not found: {PDF_PATH}")
    exit()

print("Initializing RAG...")
llm=ChatOllama(model=LLM_MODEL)
embed_model=OllamaEmbeddings(model=EMBED_MODEL)

loader=PyPDFLoader(PDF_PATH)
raw_docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
chunks=text_splitter.split_documents(raw_docs)

vector_store=Chroma.from_documents(documents=chunks, embedding=embed_model, collection_name="pdf-rag-storage")
retriever=vector_store.as_retriever(search_kwargs={"k": 3})

template="""Answer based ONLY on the context below.
Context:
{context}
Question: {question}
Answer:"""
prompt=ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain=(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

def ask_rag(query):
    print(f"(RAG) Processing: {query}")
    response=rag_chain.invoke(query)
    if hasattr(response, 'content'):
        return response.content
    return str(response)

if __name__ == "__main__":
    print(ask_rag("What is cost reduction achieved?"))

import io
import ollama
from typing import TypedDict
from langgraph.graph import StateGraph,END
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer

PDF_PATH="h&m,multimodalrag.pdf"
RETRIEVER="jinaai/jina-clip-v1"
GENERATOR="qwen2.5-vl:2b"


class AgentState(TypedDict):
    pdf_path:str
    query:str
    image_bytes:bytes
    answer:str


def retrieve(state:AgentState):
    
    #heavier than it looks. loads full PDF into RAM as images.
    #if pdf > 20 pages, this will choke - consider pagination for large documents
    pages=convert_from_path(state["pdf_path"])

    model=SentenceTransformer(RETRIEVER,trust_remote_code=True)

    page_embeddings=model.encode(pages)
    query_embedding=model.encode([state["query"]])

    best_page_idx=model.similarity(
        query_embedding,page_embeddings
    ).argmax().item()
    
    # buffer hack to get bytes without saving to disk
    # this avoids writing temporary files to disk and is more efficient

    buffer=io.BytesIO()
    pages[best_page_idx].save(buffer,format="PNG")

    return {"image_bytes":buffer.getvalue()}


def generate(state:AgentState):
    response=ollama.chat(
        model=GENERATOR,
        messages=[
            {
                "role":"user",
                "content":state["query"],
                "images":[state["image_bytes"]],
            }
        ],
    )

    return {"answer":response["message"]["content"]}


#Workflow
#simple linear graph: retrieve -> generate -> done
#we retrieve the best page and then generate the answer using vision language model 
workflow=StateGraph(AgentState)

workflow.add_node("retriever",retrieve)
workflow.add_node("generator",generate)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever","generator")
workflow.add_edge("generator",END)

app=workflow.compile()

result=app.invoke({
    "pdf_path":PDF_PATH,
    "query":"Explain the retrieval agents used",
})

print(result["answer"])

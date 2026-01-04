import base64
import os
import torch
import numpy as np
import sounddevice as sd
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from soprano import SopranoTTS
from transformers import pipeline

IMG_PATH="image2.png"

# Models
vision_llm = ChatOllama(model="qwen3-vl:2b", temperature=0)
text_llm = ChatOllama(model="gemma3:1b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
tts_model = SopranoTTS(model_name_or_path="ekwek/Soprano-80M", device=device)
SAMPLE_RATE = 32000
router_model = pipeline(
    "zero-shot-classification", 
    model="Raffix/routing_module_action_question_conversation_move_hack_debertav3_nli",
    device=0 if device == "cuda" else -1
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# Data
docs=[
    Document(page_content="The table tracks major US cities with improved demographics (2020-2023)."),
    Document(page_content="Cities with 1 million+ population: 9 total cities, 7 improved."),
    Document(page_content="Cities with 500k-1M population: 29 total cities, 27 improved."),
    Document(page_content="Cities with 250k-500k population: 53 total cities, 30 improved."),
    Document(page_content="Total cities in study: 91. Total improved: 64.")
]
vector_store=Chroma.from_documents(documents=docs, embedding=embeddings)
retriever=vector_store.as_retriever(search_kwargs={"k": 5})

class State(TypedDict):
    messages: Annotated[list, add_messages]
    router_decision: Literal["vision", "search"]
    draft_response: str
    iteration_count: int

def encode_image(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def router_node(state: State):
    question = state["messages"][-1].content
    
    #We use descriptive labels so the NLI model understands the context
    labels = ["look at image or chart", "search text documents"]
    
    # Run zero-shot classification
    results = router_model(question, candidate_labels=labels)
    
    # Get the top label
    top_label = results['labels'][0]
    
    if top_label == "look at image or chart":
        return {"router_decision": "vision"}
    return {"router_decision": "search"}

def vision_node(state: State):
    img_b64=encode_image(IMG_PATH)
    question=state["messages"][-1].content
    msg=HumanMessage(
        content=[
            {"type": "text", "text": f"Analyze this image. Question: {question}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]
    )
    response=vision_llm.invoke([msg])
    return {"messages": [response], "iteration_count": state["iteration_count"]+1}

def search_node(state: State):
    query=state["messages"][-1].content
    
    initial_docs=retriever.invoke(query)
    pairs=[[query, doc.page_content] for doc in initial_docs]
    scores=reranker.predict(pairs)
    
    ranked_docs=sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
    top_doc=ranked_docs[0][1]
    
    context=top_doc.page_content
    msg=HumanMessage(content=f"Context: {context}\nQuestion: {query}")
    response=text_llm.invoke([msg])
    return {"messages": [response], "iteration_count": state["iteration_count"]+1}

def check_consistency(state: State):
    curr=state["messages"][-1].content.strip()
    prev=state.get("draft_response", "").strip()
    if state["iteration_count"]==1: return "retry"
    if curr==prev: return "matched"
    if state["iteration_count"]>=3: return "max_retries"
    return "retry"

def update_draft(state: State):
    return {"draft_response": state["messages"][-1].content.strip()}

def tts_node(state: State):
    text = state["messages"][-1].content
    print(f"Speaking: {text}")
    
    stream = tts_model.infer_stream(text)
    audio_chunks = []
    for chunk in stream:
        audio_chunks.append(chunk)
    
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        if full_audio.dtype != np.float32:
            full_audio = full_audio.astype(np.float32)
        
        sd.play(full_audio, SAMPLE_RATE)
        sd.wait()
        
    return state

workflow=StateGraph(State)
workflow.add_node("router", router_node)
workflow.add_node("vision", vision_node)
workflow.add_node("search", search_node)
workflow.add_node("update_draft", update_draft)
workflow.add_node("tts", tts_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["router_decision"],
    {"vision": "vision", "search": "search"}
)

def verification_logic(state):
    res=check_consistency(state)
    if res in ["matched", "max_retries"]: return "tts"
    return "update_draft"

workflow.add_conditional_edges("vision", verification_logic, {"tts": "tts", "update_draft": "update_draft"})
workflow.add_conditional_edges("search", verification_logic, {"tts": "tts", "update_draft": "update_draft"})

workflow.add_conditional_edges(
    "update_draft",
    lambda state: state["router_decision"],
    {"vision": "vision", "search": "search"}
)

workflow.add_edge("tts", END)
app=workflow.compile()

inputs={
    "messages": [HumanMessage(content="How many cities have over 1 million people?")],
    "iteration_count": 0,
    "draft_response": ""
}

for event in app.stream(inputs):
    for key, value in event.items():
        if "messages" in value:
            print(f"\n{key}: {value['messages'][-1].content}")
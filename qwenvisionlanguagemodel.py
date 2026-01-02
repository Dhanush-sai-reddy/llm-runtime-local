import base64
import pyttsx3
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

engine = pyttsx3.init()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="qwen3-vl:2b", temperature=0)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vision_model(state: State):
    last_message = state["messages"][-1]
    image_data = encode_image("image2.png")
    multimodal_message = HumanMessage(
        content=[
            {"type": "text", "text": last_message.content},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
    )
    response = llm.invoke([multimodal_message])
    return {"messages": [response]}

def speak_output(state: State):
    text = state["messages"][-1].content
    engine.say(text)
    engine.runAndWait()
    return state

builder = StateGraph(State)
builder.add_node("vision_node", call_vision_model)
builder.add_node("tts_node", speak_output)

builder.add_edge(START, "vision_node")
builder.add_edge("vision_node", "tts_node")
builder.add_edge("tts_node", END)

graph = builder.compile()

inputs = {
    "messages": [HumanMessage(content="What is in this image? How many cities have more than 1 million people?")]
}

for event in graph.stream(inputs):
    for key, value in event.items():
        print(f"\nResponse from {key}:")
        print(value["messages"][-1].content)
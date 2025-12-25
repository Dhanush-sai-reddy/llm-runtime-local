import json
import sys
from langchain_ollama import ChatOllama

try:
    from localrag import ask_rag
    from llmforsql import ask_sql
except ImportError as e:
    print(f"Error: {e}")
    sys.exit()

try:
    llm=ChatOllama(model="gemma3:1b", format="json")
except TypeError:
    llm=ChatOllama(model="gemma3:1b")

ROUTER_PROMPT="""
You are a router. Return JSON only.
Tools:
1. "SQL": For database (Music, Sales, Customers).
2. "RAG": For documents (PDF, FLAP-SAM).

Question: {question}
JSON: {{ "tool": "SQL" }} or {{ "tool": "RAG" }}
"""

def clean_json(text):
    text=text.strip()
    if text.startswith("```json"): text=text[7:]
    if text.startswith("```"): text=text[3:]
    if text.endswith("```"): text=text[:-3]
    return text.strip()

def main():
    while True:
        user_input=input("\nYou: ")
        if user_input.lower() in ["q", "exit"]: break
        
        try:
            raw_response=llm.invoke(ROUTER_PROMPT.format(question=user_input)).content
            decision=json.loads(clean_json(raw_response))
            tool=decision.get("tool")
            
            if tool=="SQL":
                print(ask_sql(user_input))
            elif tool=="RAG":
                print(ask_rag(user_input))
            else:
                print("Unknown tool")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__=="__main__":
    main()
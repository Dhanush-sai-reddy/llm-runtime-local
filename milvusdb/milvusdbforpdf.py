import ollama
from pymilvus import MilvusClient
from pypdf import PdfReader

client = MilvusClient("local_rag.db")
if client.has_collection("handbook"): client.drop_collection("handbook")
client.create_collection("handbook", dimension=2048)

reader = PdfReader("employeehandbook.pdf")
docs = [p.extract_text() for p in reader.pages if p.extract_text()]
data = []

for i, txt in enumerate(docs):
    vec = ollama.embeddings(model="gemma3:1b", prompt=txt)["embedding"]
    data.append({"id": i, "vector": vec, "text": txt})

if data: client.insert("handbook", data)

q = "No of days as casual leave?"
q_vec = ollama.embeddings(model="gemma3:1b", prompt=q)["embedding"]

res = client.search("handbook", [q_vec], limit=2, output_fields=["text"])
ctx = "\n".join([r["entity"]["text"] for r in res[0]])

print(ollama.chat("gemma3:1b", [{"role": "user", "content": f"Context: {ctx}\nQuestion: {q}"}])["message"]["content"])
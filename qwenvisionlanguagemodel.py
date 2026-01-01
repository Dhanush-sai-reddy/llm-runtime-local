import ollama
#image as bytes
with open("image2.png", "rb") as f:
    img = f.read()

print("Thinking...")

response = ollama.chat(
    model="qwen3-vl:2b",
    messages=[
        {
            "role": "user",
            "content": "What is in this image?how many cities have more than 1 million people?",
            "images": [img],
        }
    ],
)

print(response["message"]["content"])

import base64
import requests
from huggingface_hub import InferenceClient

API_KEY = "hf_YOUR_API_KEY_HERE"

client = InferenceClient(api_key=API_KEY)

with open("image2.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.chat_completion(
    model="Qwen/Qwen2-VL-2B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image? How many cities have more than 1 million people?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        }
    ],
    max_tokens=500
)

text_content = response.choices[0].message.content
print(text_content)

API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
headers = {"Authorization": f"Bearer {API_KEY}"}

tts_response = requests.post(API_URL, headers=headers, json={"inputs": text_content})

if tts_response.status_code == 200:
    with open("output.flac", "wb") as f:
        f.write(tts_response.content)
else:
    print(tts_response.text)
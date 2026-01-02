import torch
import numpy as np
import copy
import warnings
from typing import TypedDict, Any, List
from langgraph.graph import StateGraph, END
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image

warnings.filterwarnings("ignore")

# Global Load - model loaded once at startup-(not efficient)
PRETRAINED = "model_zoo/BBBBCHAN/LLaVA-Scissor-baseline-0.5B"
MODEL_NAME = "llava_qwen"
DEVICE = "cuda"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    PRETRAINED, None, MODEL_NAME, device_map="auto", attn_implementation="sdpa"
)
model.eval()

class VideoState(TypedDict):
    video_path: str
    query: str
    frames_tensor: Any 
    image_sizes: List  
    response: str

def load_video_node(state: VideoState):
    path = state['video_path']
    max_frames = 16 #crucial to select right number of frames to balance quality and memory
    
    try:
        vr = VideoReader(path, ctx=cpu(0))
    except Exception:
        return {"response": "Error: Video file is trash or missing."}

    total = len(vr)
    
    indices = np.linspace(0, total - 1, max_frames, dtype=int).tolist()
    raw_frames = vr.get_batch(indices).asnumpy() 
    
    proc_frames = image_processor.preprocess(raw_frames, return_tensors="pt")["pixel_values"].half().to(DEVICE)
    
    sizes = [(raw_frames.shape[2], raw_frames.shape[1]) for _ in range(max_frames)]

    return {
        "frames_tensor": [proc_frames], 
        "image_sizes": sizes
    }

def inference_node(state: VideoState):
    if state.get("response"): 
        return 

    conv = copy.deepcopy(conv_templates["qwen_2"])
    q = f"{DEFAULT_IMAGE_TOKEN}\n{state['query']}"
    
    conv.append_message(conv.roles[0], q)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
    
    out = model.generate(
        input_ids,
        images=state['frames_tensor'],
        image_sizes=state['image_sizes'],
        do_sample=False, 
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"],
    )
    
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return {"response": decoded}

# Graph
workflow = StateGraph(VideoState)

workflow.add_node("loader", load_video_node)
workflow.add_node("captioner", inference_node)

workflow.set_entry_point("loader")
workflow.add_edge("loader", "captioner")
workflow.add_edge("captioner", END)

app = workflow.compile()

# Test
if __name__ == "__main__":
    result = app.invoke({
        "video_path": "test_video.mp4",
        "query": "Describe this video."
    })
    
    print(f"OUTPUT: {result['response']}")
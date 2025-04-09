from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

# Load the LLaVA model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

def llava_generate(image: Image.Image, prompt: str) -> str:
    """
    Generate a response from LLaVA given an image and a prompt.
    """
    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    
    # Decode and return the generated text
    response = processor.decode(output[0], skip_special_tokens=True)
    return response

def llava_caption(image: Image.Image) -> str:
    """
    Generate a caption for the given image.
    """
    prompt = f"USER: <image>\nDescribe the content of this image. ASSISTANT:"
    return llava_generate(image, prompt)

def llava_answer(image: Image.Image, question: str) -> str:
    """
    Answer a question about the given image.
    """
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    return llava_generate(image, prompt)

def llava_tags(image: Image.Image) -> list[str]:
    """
    Generate descriptive tags for the given image.
    """
    
    prompt = f"USER: <image>\nList keywords that describe this image. ASSISTANT:"
    response = llava_generate(image, prompt)
    tags = [tag.strip() for tag in response.replace(".", "").split(",")]
    return tags

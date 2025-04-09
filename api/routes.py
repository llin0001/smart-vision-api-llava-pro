from fastapi import APIRouter, UploadFile, File, Form
from llava_inference.predictor import llava_caption, llava_answer, llava_tags
from PIL import Image
import io

router = APIRouter()

@router.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return {"caption": llava_caption(image)}

@router.post("/vqa")
async def vqa_image(file: UploadFile = File(...), question: str = Form(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return {"answer": llava_answer(image, question)}

@router.post("/tags")
async def tag_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return {"tags": llava_tags(image)}

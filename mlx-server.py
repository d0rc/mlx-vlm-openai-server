from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import base64
import tempfile
import os

app = FastAPI(title="Image Processing Service")

# Load model at startup
MODEL_PATH = "mlx-community/Qwen2-VL-7B-Instruct-8bit"
model, processor = load(MODEL_PATH)
config = load_config(MODEL_PATH)

# Request models
class ImageURL(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

class Message(BaseModel):
    role: str
    content: List[ContentItem]

class OpenAIRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None

# Response models
class ChatCompletionMessage(BaseModel):
    content: str

class ChatCompletionChoice(BaseModel):
    message: ChatCompletionMessage

class ErrorMessage(BaseModel):
    message: str

class OpenAIResponse(BaseModel):
    choices: List[ChatCompletionChoice] = []
    error: Optional[ErrorMessage] = None

def decode_image(data_url: str) -> bytes:
    if ',' not in data_url:
        raise ValueError("Invalid data URL")
    header, data = data_url.split(',', 1)
    return base64.b64decode(data)

@app.post("/v1/chat/completions", response_model=OpenAIResponse)
async def chat_completions(request: OpenAIRequest):
    temp_files = []
    try:
        # Extract user messages
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return OpenAIResponse(error=ErrorMessage(message="No user message found"))

        prompt_parts = []
        temp_files = []

        for message in user_messages:
            for content in message.content:
                if content.type == "text" and content.text:
                    prompt_parts.append(content.text)
                elif content.type == "image_url" and content.image_url:
                    try:
                        image_data = decode_image(content.image_url.url)
                        suffix = ".png" if "image/png" in content.image_url.url else ".jpg"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                            f.write(image_data)
                            temp_files.append(f.name)
                    except Exception as e:
                        for f in temp_files:
                            try: os.remove(f)
                            except: pass
                        return OpenAIResponse(error=ErrorMessage(message=f"Image processing failed: {str(e)}"))

        prompt = " ".join(prompt_parts)
        if not prompt:
            return OpenAIResponse(error=ErrorMessage(message="No text prompt provided"))
        if not temp_files:
            return OpenAIResponse(error=ErrorMessage(message="No images provided"))

        # Process images and generate response
        formatted_prompt = apply_chat_template(
            processor,
            config,
            prompt,
            num_images=len(temp_files))
        
        output = generate(
            model,
            processor,
            temp_files,
            formatted_prompt,
            verbose=True,
            max_tokens=request.max_tokens or 1024,
            temp=0.6
        )

        # Cleanup temp files
        for f in temp_files:
            try: os.remove(f)
            except: pass

        return OpenAIResponse(choices=[
            ChatCompletionChoice(message=ChatCompletionMessage(content=output))
        ])

    except Exception as e:
        # Cleanup temp files on error
        for f in temp_files:
            try: os.remove(f)
            except: pass
        return OpenAIResponse(error=ErrorMessage(message=str(e)))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

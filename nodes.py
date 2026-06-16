import io
import os
import base64
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Try to import the new OpenAI library, fall back to old style
try:
    import openai
    # Check if we have the new style (v1.0.0+)
    if hasattr(openai, 'OpenAI'):
        USE_NEW_OPENAI = True
    else:
        USE_NEW_OPENAI = False
except ImportError:
    raise ImportError("Please install openai: pip install openai")

load_dotenv()

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo"
]

class OpenAICaptionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {}),
                "model": (MODELS,),
                # New fields for local server support
                "api_url": ("STRING", {"default": "", "placeholder": "http://localhost:8080/v1"}),
                "api_key": ("STRING", {"default": "", "password": True}),
                # Multiline prompts for larger text areas
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "caption_prompt": ("STRING", {"default": "What's in this image?", "multiline": True}),
                "max_tokens": ("INT", {"default": 300}),
                "temperature": ("FLOAT", {"default": 0.5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_out",)
    CATEGORY = "openai"
    FUNCTION = "caption"

    def caption(self, image_in, model, api_url, api_key,
                system_prompt, caption_prompt, max_tokens, temperature):
        # Convert tensor to PIL Image
        pil_image = Image.fromarray(
            np.clip(255. * image_in.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Determine API key: use provided, fallback to env var
        effective_key = api_key.strip() if api_key.strip() else os.getenv("OPENAI_API_KEY")
        effective_url = api_url.strip() if api_url.strip() else None

        if USE_NEW_OPENAI:
            # New OpenAI library (v1.0.0+)
            client_kwargs = {}
            if effective_url:
                client_kwargs["base_url"] = effective_url
            if effective_key:
                client_kwargs["api_key"] = effective_key
            client = openai.OpenAI(**client_kwargs)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            caption = response.choices[0].message.content.strip()
            
        else:
            # Old OpenAI library (pre-v1.0.0)
            if effective_key:
                openai.api_key = effective_key
            if effective_url:
                openai.api_base = effective_url
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            caption = response.choices[0].message.content.strip()

        if caption is None:
            raise ValueError("No content in response")
            
        caption = caption.replace("json", "").replace("```", "").strip()

        return (caption,)

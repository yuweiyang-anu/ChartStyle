"""Nano Banana Pro (Google Gemini image) style-transfer backend.

Uses the public ``google-genai`` SDK with a plain API key (``GOOGLE_API_KEY``)
— no internal proxy or service-account flow. Defaults to ``Nano Banana Pro``
(``gemini-3-pro-image-preview``); pass ``model_name`` to use a different image
model (e.g. ``gemini-2.5-flash-image-preview`` for the original Nano Banana).

The style reference and content image are sent as two image parts after the
generation prompt; the first generated image part in the response is returned.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from typing import Any, Dict, Optional, Tuple

from google import genai
from google.genai import types as genai_types
from PIL import Image

from models import image_utils
from models.base import StyleTransferModel
from prompts.generation_prompts import STYLE_TRANSFER_PROMPT


class NanoBananaProModel(StyleTransferModel):
    """Gemini image-edit model accessed via a standard Google API key."""

    def __init__(
        self,
        name: str = "nano_banana_pro",
        model_name: str = "gemini-3-pro-image-preview",
        api_key_env: str = "GOOGLE_API_KEY",
        api_key: Optional[str] = None,
    ):
        super().__init__(name)
        self.model_name = model_name
        self.system_prompt = STYLE_TRANSFER_PROMPT
        key = api_key or os.getenv(api_key_env)
        if not key:
            raise EnvironmentError(
                f"Google API key not found. Set ${api_key_env} or pass api_key=..."
            )
        self.client = genai.Client(api_key=key)

    @staticmethod
    def _image_to_png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _extract_image(result: Any) -> Image.Image:
        for candidate in getattr(result, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if data:
                    return Image.open(io.BytesIO(data)).convert("RGB")
        raise RuntimeError("Gemini response did not contain image data")

    async def generate(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        **kwargs: Any,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        style_rgb = image_utils.alpha_over_white(style_reference)
        content_rgb = image_utils.alpha_over_white(content_input)
        start = time.time()

        prompt_part = genai_types.Part.from_text(text=self.system_prompt.strip())
        style_part = genai_types.Part.from_bytes(
            data=self._image_to_png_bytes(style_rgb), mime_type="image/png"
        )
        content_part = genai_types.Part.from_bytes(
            data=self._image_to_png_bytes(content_rgb), mime_type="image/png"
        )

        def call_gemini() -> Any:
            return self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[prompt_part, style_part, content_part],
                    )
                ],
            )

        max_retries = kwargs.get("max_retries", 3)
        delay_seconds = kwargs.get("retry_delay_seconds", 5)
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                result = await asyncio.to_thread(call_gemini)
                generated_image = self._extract_image(result)
                metadata = {
                    "model_name": self.name,
                    "gemini_model": self.model_name,
                    "prompt": self.system_prompt.strip(),
                    "generation_time": time.time() - start,
                }
                return generated_image, metadata
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    print(f"[Gemini] attempt {attempt + 1} failed: {exc}; retrying in {delay_seconds}s")
                    await asyncio.sleep(delay_seconds)

        raise RuntimeError(
            f"Gemini image generation failed after {max_retries} attempts. Last error: {last_exc}"
        ) from last_exc

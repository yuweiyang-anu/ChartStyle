"""OpenAI gpt-image-* style-transfer backend (images.edit API).

Supports any of the OpenAI image-edit models — ``gpt-image-1`` , ``gpt-image-1.5``, ``gpt-image-2`` — selected via ``model_name``. Both
the style reference and the content image are uploaded to ``images.edit`` and the
generation prompt instructs the model to redraw the content in the reference's
style. Requires ``OPENAI_API_KEY`` (or an explicit ``api_key``).
"""

from __future__ import annotations

import base64
import io
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import openai
import pydantic
from loguru import logger
from PIL import Image

from models.base import StyleTransferModel
from models.image_utils import alpha_over_white
from prompts.generation_prompts import STYLE_TRANSFER_PROMPT


class OpenAIImageEditConfig(pydantic.BaseModel):
    OPENAI_MODEL: str = "gpt-image-1"
    QUALITY: str = "medium"
    BACKGROUND: str = "auto"
    SYSTEM_PROMPT: str = STYLE_TRANSFER_PROMPT


class OpenAIStyleTransferModel(StyleTransferModel):
    """gpt-image-* image-edit pipeline."""

    def __init__(self, model_name: str = "gpt-image-1", api_key: Optional[str] = None):
        super().__init__(f"openai_{model_name}")
        self.model_name = model_name
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.config = OpenAIImageEditConfig(OPENAI_MODEL=model_name)

    @staticmethod
    def _generate_image_size(pil_image: Image.Image) -> str:
        """Snap the content aspect ratio to one of OpenAI's three supported sizes."""
        w, h = pil_image.size
        ratio = w / h
        square = abs(ratio - 1.0)              # 1024x1024
        landscape = abs(ratio - (1536 / 1024))  # 1536x1024
        portrait = abs(ratio - (1024 / 1536))   # 1024x1536
        smallest = min(square, landscape, portrait)
        if smallest == square:
            return "1024x1024"
        if smallest == landscape:
            return "1536x1024"
        return "1024x1536"

    async def generate(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        **kwargs: Any,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        start = time.time()
        prompt = self.config.SYSTEM_PROMPT
        quality = self.config.QUALITY
        background = self.config.BACKGROUND

        try:
            style_reference = alpha_over_white(style_reference)
            content_input = alpha_over_white(content_input)
            with tempfile.NamedTemporaryFile(suffix=".png") as style_tmp, \
                    tempfile.NamedTemporaryFile(suffix=".png") as content_tmp:
                style_reference.save(style_tmp.name, format="PNG")
                content_input.save(content_tmp.name, format="PNG")
                size = self._generate_image_size(content_input)

                result = await self.client.images.edit(
                    model=self.model_name,
                    image=[open(style_tmp.name, "rb"), open(content_tmp.name, "rb")],
                    prompt=prompt,
                    quality=quality,
                    size=size,
                    background=background,
                )

            if not result.data or result.data[0].b64_json is None:
                raise RuntimeError("No image returned from OpenAI")

            image_bytes = base64.b64decode(result.data[0].b64_json)
            generated_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            metadata = {
                "model_name": self.model_name,
                "image_size": size,
                "quality": quality,
                "background": background,
                "prompt": prompt,
                "generation_time": time.time() - start,
            }
            return generated_image, metadata

        except Exception as exc:
            logger.error(f"OpenAI API error: {exc}")
            # Fall back to the unmodified content image so the run can continue;
            # the scorer harness records the error in metadata.
            return content_input, {
                "model_name": self.model_name,
                "error": str(exc),
                "generation_time": time.time() - start,
            }

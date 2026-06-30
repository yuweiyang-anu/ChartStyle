"""Common interface every style-transfer inference backend implements."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from PIL import Image


class StyleTransferModel:
    """Abstract base class for a chart style-transfer model.

    A model takes a *style reference* image and a *content* image and produces a
    restyled output: the content's data/structure rendered in the reference's
    visual style. Implementations return the generated image plus a metadata
    dict (timings, model name, prompt, raw usage, etc.).
    """

    def __init__(self, name: str):
        self.name = name

    async def generate(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        **kwargs: Any,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generate one restyled image and return (image, metadata)."""
        raise NotImplementedError

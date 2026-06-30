"""CLIP-based image-similarity scorers (non-LLM benchmark metrics).

Two reference-free signals computed with OpenAI CLIP (ViT-B/32 by default):

  * SemanticConsistencyScorer    -> Semantic Consistency : compares CLIP embeddings
        of the content input and the generated chart to ensure the original
        visualization's semantic structure is preserved despite the style transfer.
  * CLIPStylisticFidelityScorer  -> CLIP Stylistic Fidelity : projects the style
        reference and generated chart through a CLIP encoder and reports their
        cosine similarity (texture, palette, and mark rendering alignment).

Both are in [-1, 1] (higher is better). CLIP weights download automatically on
first use of ``clip.load``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import clip
from PIL import Image

from scorers.base import BaseScorer


class _CLIPBackbone:
    """Lazily-loaded, process-wide shared CLIP model keyed by model name."""

    _cache: Dict[str, Any] = {}

    @classmethod
    def get(cls, model_name: str):
        if model_name not in cls._cache:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(model_name, device=device)
            cls._cache[model_name] = (model, preprocess, device)
        return cls._cache[model_name]


class _BaseCLIPScorer(BaseScorer):
    def __init__(self, name: str, model_name: str = "ViT-B/32"):
        super().__init__(name)
        self.model, self.preprocess, self.device = _CLIPBackbone.get(model_name)

    def _encode(self, image: Image.Image) -> torch.Tensor:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(tensor)

    def _cosine(self, a: Image.Image, b: Image.Image) -> float:
        sim = F.cosine_similarity(self._encode(a), self._encode(b), dim=1)
        return float(sim.cpu().item())


class CLIPStylisticFidelityScorer(_BaseCLIPScorer):
    """CLIP Stylistic Fidelity: CLIP cosine similarity (style reference vs. output)."""

    def __init__(self, model_name: str = "ViT-B/32"):
        super().__init__("clip_stylistic_fidelity", model_name)

    async def score(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        generated_output: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        return self._cosine(style_reference, generated_output)


class SemanticConsistencyScorer(_BaseCLIPScorer):
    """Semantic Consistency: CLIP cosine similarity (content input vs. output)."""

    def __init__(self, model_name: str = "ViT-B/32"):
        super().__init__("semantic_consistency", model_name)

    async def score(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        generated_output: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        return self._cosine(content_input, generated_output)

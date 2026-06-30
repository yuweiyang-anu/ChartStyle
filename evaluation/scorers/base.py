"""Scorer interface for local (weave-free) evaluation.

Every scorer exposes the same async API:

    score(style_reference, content_input, generated_output, metadata) -> float | bool

and, optionally, ``score_with_reasoning`` which additionally returns the judge's
textual justification. Scores are plain Python numbers/booleans — there is no
tracing, logging, or remote upload of any kind.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image


class BaseScorer:
    """Base class for all evaluation scorers."""

    def __init__(self, name: str):
        self.name = name

    async def score(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        generated_output: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        raise NotImplementedError

    async def score_with_reasoning(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        generated_output: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        value = await self.score(style_reference, content_input, generated_output, metadata)
        return {"score": value, "reasoning": None}

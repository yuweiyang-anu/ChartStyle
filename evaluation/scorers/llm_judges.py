"""LLM-as-judge scorers (GPT-4o) for the ChartStyle-Bench metrics.

These back the "LLM Evaluation" metrics of the paper (GPT-4o is the judge for all):

  * LLMContentJudgeScorer    -> Content Consistency : how well the generated chart
        preserves the original content, 1-5 (higher is better).
  * LLMStyleJudgeScorer      -> Style Similarity    : how closely the generated chart
        adheres to the reference's visual style, 1-5 (higher is better).
  * LLMContentLeakageScorer  -> Content Leakage     : 1 if unintended visual elements
        from the reference appear in the output, 0 otherwise (lower is better).

The Overall Score is derived in the harness (harmonic mean of Content Consistency
and Style Similarity, with Style Similarity set to 1 when leakage is detected) —
there is no separate LLM judge for it.

Each judge calls ``chat.completions.parse`` with a Pydantic ``response_format`` so
the score and reasoning come back as structured data. Needs ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

import openai
import pydantic
from loguru import logger
from PIL import Image

from models import image_utils
from prompts.judge_prompts import (
    CONTENT_JUDGE_PROMPT,
    STYLE_JUDGE_PROMPT,
    CONTENT_LEAKAGE_PROMPT,
)
from scorers.base import BaseScorer

# GPT-4o is the judge used for all LLM-based metrics in the paper.
DEFAULT_JUDGE_MODEL = "gpt-4o"


class ReasoningScoreOutput(pydantic.BaseModel):
    reasoning: str
    score: float


class BooleanCheck(pydantic.BaseModel):
    reasoning: str
    outcome: bool


def _pil_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _image_part(image: Image.Image) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_pil_to_b64(image)}"},
    }


class _BaseLLMJudge(BaseScorer):
    def __init__(self, name: str, model: str = DEFAULT_JUDGE_MODEL, api_key: Optional[str] = None):
        super().__init__(name)
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key, timeout=90.0, max_retries=2)


class LLMContentJudgeScorer(_BaseLLMJudge):
    """Content Consistency: content preservation, 1-5 (content input vs. output)."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, api_key: Optional[str] = None):
        super().__init__("content_consistency", model, api_key)

    async def score_with_reasoning(self, style_reference, content_input, generated_output, metadata=None):
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": CONTENT_JUDGE_PROMPT},
                    {"type": "text", "text": "Content Input:"},
                    _image_part(content_input),
                    {"type": "text", "text": "Generated Output:"},
                    _image_part(generated_output),
                ],
            }],
            response_format=ReasoningScoreOutput,
        )
        parsed = response.choices[0].message.parsed
        if not isinstance(parsed, ReasoningScoreOutput):
            logger.warning("LLMContentJudgeScorer: failed to parse response")
            return {"score": float("nan"), "reasoning": "LLM response parsing failed"}
        return parsed.model_dump()

    async def score(self, style_reference, content_input, generated_output, metadata=None) -> float:
        result = await self.score_with_reasoning(style_reference, content_input, generated_output, metadata)
        return float(result["score"])


class LLMStyleJudgeScorer(_BaseLLMJudge):
    """Style Similarity: adherence to the reference style, 1-5 (style ref vs. output)."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, api_key: Optional[str] = None):
        super().__init__("style_similarity", model, api_key)

    async def score_with_reasoning(self, style_reference, content_input, generated_output, metadata=None):
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": STYLE_JUDGE_PROMPT},
                    {"type": "text", "text": "Style Reference:"},
                    _image_part(style_reference),
                    {"type": "text", "text": "Generated Output:"},
                    _image_part(generated_output),
                ],
            }],
            response_format=ReasoningScoreOutput,
        )
        parsed = response.choices[0].message.parsed
        if not isinstance(parsed, ReasoningScoreOutput):
            logger.warning("LLMStyleJudgeScorer: failed to parse response")
            return {"score": float("nan"), "reasoning": "LLM response parsing failed"}
        return parsed.model_dump()

    async def score(self, style_reference, content_input, generated_output, metadata=None) -> float:
        result = await self.score_with_reasoning(style_reference, content_input, generated_output, metadata)
        return float(result["score"])


class LLMContentLeakageScorer(_BaseLLMJudge):
    """Content Leakage: 1 if reference content leaks into the output, else 0 (lower better)."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, api_key: Optional[str] = None):
        super().__init__("content_leakage", model, api_key)

    def _b64(self, image: Image.Image) -> Dict[str, Any]:
        # Leakage inspection keeps transparency visible via a checkerboard composite.
        return _image_part(image_utils.alpha_over_checker(image))

    async def score_with_reasoning(self, style_reference, content_input, generated_output, metadata=None):
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": CONTENT_LEAKAGE_PROMPT},
                    {"type": "text", "text": "Style Reference:"},
                    self._b64(style_reference),
                    {"type": "text", "text": "Generated Output:"},
                    self._b64(generated_output),
                ],
            }],
            response_format=BooleanCheck,
        )
        parsed = response.choices[0].message.parsed
        if not isinstance(parsed, BooleanCheck):
            logger.warning("LLMContentLeakageScorer: failed to parse response")
            return {"outcome": False, "reasoning": "LLM response parsing failed"}
        return parsed.model_dump()

    async def score(self, style_reference, content_input, generated_output, metadata=None) -> bool:
        result = await self.score_with_reasoning(style_reference, content_input, generated_output, metadata)
        return bool(result["outcome"])

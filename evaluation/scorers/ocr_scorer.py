"""PaddleOCR text-overlap scorer — the OCRScore benchmark metric.

Measures how faithfully textual annotations and labels are retained between the
content image and the generated output. Both images are OCR'd with PaddleOCR,
tokens are normalised to lowercase alphanumerics, and the word-level F1 over the
token multisets is returned in [0, 1] (1.0 = identical text, 0.0 = no overlap;
higher is better).

If PaddleOCR is unavailable the scorer degrades gracefully and returns 0.5.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger
from PIL import Image

from models import image_utils
from scorers.base import BaseScorer

try:
    from paddleocr import PaddleOCR
    _PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    _PADDLEOCR_AVAILABLE = False

# One OCR engine per language, shared across all scorer instances.
_OCR_INSTANCES: Dict[str, Any] = {}


class PaddleOCROverlapScorer(BaseScorer):
    """Token-level F1 overlap between content and generated text (via PaddleOCR)."""

    def __init__(self, lang: str = "en", confidence_threshold: float = 0.6):
        super().__init__("ocr_score")
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self.available = _PADDLEOCR_AVAILABLE
        self.ocr = None

        if not self.available:
            logger.warning("PaddleOCR not installed; PaddleOCROverlapScorer returns 0.5.")
            return

        if lang not in _OCR_INSTANCES:
            try:
                _OCR_INSTANCES[lang] = PaddleOCR(use_angle_cls=True, lang=lang)
            except Exception as exc:
                logger.error(f"Failed to initialize PaddleOCR: {exc}")
                self.available = False
                return
        self.ocr = _OCR_INSTANCES[lang]

    @staticmethod
    def _normalize_to_words(text: str) -> list[str]:
        normalized = re.sub(r"[^0-9a-z]+", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized.split(" ") if normalized else []

    def _iter_detection_entries(self, results: Any):
        if isinstance(results, (list, tuple)):
            if (
                len(results) == 2
                and isinstance(results[0], (list, tuple))
                and isinstance(results[1], (list, tuple))
                and len(results[1]) >= 2
                and isinstance(results[1][0], str)
            ):
                yield results
            else:
                for item in results:
                    yield from self._iter_detection_entries(item)

    def _word_counter(self, image: Image.Image) -> Counter:
        if not self.available or self.ocr is None:
            return Counter()
        prepared = image_utils.alpha_over_checker(image)
        bgr = np.array(prepared.convert("RGB"))[:, :, ::-1]
        try:
            results = self.ocr.ocr(bgr, cls=True)
        except Exception as exc:
            logger.error(f"PaddleOCR inference failed: {exc}")
            return Counter()

        words: list[str] = []
        for entry in self._iter_detection_entries(results or []):
            text_info = entry[1]
            if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                continue
            text, score = text_info[0], text_info[1]
            if not isinstance(text, str):
                continue
            if score is not None and score < self.confidence_threshold:
                continue
            words.extend(self._normalize_to_words(text))
        return Counter(words)

    @staticmethod
    def _f1(content: Counter, output: Counter) -> float:
        if not content and not output:
            return 1.0
        total_content = sum(content.values())
        total_output = sum(output.values())
        if total_content == 0 or total_output == 0:
            return 0.0
        matched = sum(min(content[w], output.get(w, 0)) for w in content)
        if matched == 0:
            return 0.0
        recall = matched / total_content
        precision = matched / total_output
        if precision + recall == 0:
            return 0.0
        return float(max(0.0, min(1.0, (2 * precision * recall) / (precision + recall))))

    async def score(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        generated_output: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not self.available or self.ocr is None:
            return 0.5
        return self._f1(self._word_counter(content_input), self._word_counter(generated_output))

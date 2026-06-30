"""Local evaluation harness: generate, score, aggregate, and save — no remote upload.

For each benchmark sample the harness (1) runs the model to produce a restyled
image, (2) runs every configured scorer, and (3) derives the paper's Overall
Score — the harmonic mean of Content Consistency and Style Similarity, with Style
Similarity set to 1 when content leakage is detected. Per-sample results, generated
images, and an aggregate summary are written to disk as JSON and PNG files.
Nothing is logged to Weave/W&B or any external service.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image
from tqdm import tqdm

from dataset import BenchSample
from models.base import StyleTransferModel
from scorers.base import BaseScorer

# Paper metric keys (scorer.name values) that feed the derived Overall Score.
CONTENT_CONSISTENCY = "content_consistency"
STYLE_SIMILARITY = "style_similarity"
CONTENT_LEAKAGE = "content_leakage"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))


def _overall_score(scores: Dict[str, Any]) -> float:
    """Overall Score: harmonic mean of Content Consistency & Style Similarity;
    Style Similarity is set to 1 when content leakage is detected."""
    content = scores.get(CONTENT_CONSISTENCY)
    style = scores.get(STYLE_SIMILARITY)
    leakage = scores.get(CONTENT_LEAKAGE)

    if not _is_number(content) or not _is_number(style):
        return float("nan")
    if isinstance(leakage, bool) and leakage:
        style = 1.0
    if content <= 0 or style <= 0:
        return 0.0
    return 2.0 / ((1.0 / content) + (1.0 / style))


class EvaluationHarness:
    def __init__(
        self,
        model: StyleTransferModel,
        scorers: List[BaseScorer],
        output_dir: str = "eval_results",
        concurrency: int = 4,
        save_images: bool = True,
    ):
        self.model = model
        self.scorers = scorers
        self.concurrency = concurrency
        self.save_images = save_images
        self.run_dir = Path(output_dir) / f"{model.name}_{datetime.now():%Y%m%d_%H%M%S}"
        self.images_dir = self.run_dir / "images"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

    async def _score_one(
        self,
        style: Image.Image,
        content: Image.Image,
        generated: Image.Image,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        async def run(scorer: BaseScorer):
            try:
                return scorer.name, await scorer.score_with_reasoning(style, content, generated, metadata)
            except Exception as exc:
                logger.error(f"Scorer {scorer.name} failed: {exc}")
                return scorer.name, {"score": float("nan"), "reasoning": f"error: {exc}"}

        payloads = await asyncio.gather(*(run(s) for s in self.scorers))
        scores: Dict[str, Any] = {}
        reasonings: Dict[str, Any] = {}
        for name, payload in payloads:
            value = payload.get("score", payload.get("outcome"))
            scores[name] = value
            if payload.get("reasoning") is not None:
                reasonings[name] = payload["reasoning"]
        scores["overall_score"] = _overall_score(scores)
        return {"scores": scores, "reasonings": reasonings}

    async def _process(self, sample: BenchSample, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        async with semaphore:
            style = sample.style_reference
            content = sample.content_image
            try:
                generated, gen_meta = await self.model.generate(style, content)
            except Exception as exc:
                logger.error(f"[{sample.sample_id}] generation failed: {exc}")
                return {
                    "sample_id": sample.sample_id,
                    "content_type": sample.content_type,
                    "error": str(exc),
                    "scores": {},
                }

            if self.save_images and isinstance(generated, Image.Image):
                generated.save(self.images_dir / f"{sample.sample_id}.png")

            scored = await self._score_one(style, content, generated, gen_meta)
            return {
                "sample_id": sample.sample_id,
                "content_type": sample.content_type,
                "generation_metadata": {k: v for k, v in gen_meta.items() if k != "prompt"},
                **scored,
            }

    async def run(self, samples: List[BenchSample]) -> Dict[str, Any]:
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [asyncio.ensure_future(self._process(s, semaphore)) for s in samples]

        records: List[Dict[str, Any]] = []
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {self.model.name}"):
            records.append(await fut)
        records.sort(key=lambda r: r["sample_id"])

        summary = self._aggregate(records)
        self._save(records, summary)
        return summary

    def _aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        metric_names = {name for r in records for name in r.get("scores", {})}
        summary: Dict[str, Any] = {
            "model_name": self.model.name,
            "total_samples": len(records),
            "failed_generations": sum(1 for r in records if r.get("error")),
            "metrics": {},
        }
        for name in sorted(metric_names):
            values = [r["scores"][name] for r in records if name in r.get("scores", {})]
            bools = [v for v in values if isinstance(v, bool)]
            numbers = [float(v) for v in values if _is_number(v) and not isinstance(v, bool)]
            if bools and not numbers:  # boolean metric (e.g. leakage)
                summary["metrics"][name] = {"rate_true": sum(bools) / len(bools), "count": len(bools)}
            elif numbers:
                summary["metrics"][name] = {
                    "mean": sum(numbers) / len(numbers),
                    "min": min(numbers),
                    "max": max(numbers),
                    "count": len(numbers),
                }
        return summary

    def _save(self, records: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        with open(self.run_dir / "results.json", "w") as f:
            json.dump(records, f, indent=2, default=str)
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Results written to {self.run_dir}")

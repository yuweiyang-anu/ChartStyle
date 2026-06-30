# -*- coding: utf-8 -*-
"""
ChartForge — Stage IV, step 1 of 3: LLM multi-dimensional quality scoring.

This is the LLM-as-judge component of Stage IV. Each candidate triplet is scored
by GPT-4o along three independent axes — visual/technical quality, content
preservation (strict, character-level), and style similarity — on a 1-5 scale.
The scores feed the OCR F1 check (``stage4b_ocr_filtering.py``) and the final
threshold aggregation (``stage4c_aggregate_filter.py``).

The three judge prompts are kept verbatim in ``prompts/evaluation.py`` (and match
the paper's supplementary material); they are imported below rather than inlined
so there is a single auditable source of truth.

----------------------------------------------------------------------
Evaluation module: LLM-only evaluation for charts and FDT (FDT = flowcharts, diagrams, tables).
Deleted all OCR-related scoring and dependencies. Provides:
- LLMVisualQualityJudgeScorer  (image-only; charts & FDT-aware; strong cropping checks)
- LLMContentJudgeScorer        (STRICT: character-level content preservation)
- LLMStyleJudgeScorer          (optimized style prompt)

Features:
- Dataset-level tqdm progress bar (main process only; no per-item/thread prints)
- JSONL/CLI support
- Per-item JSON outputs + overall JSON

Notes:
- Requires OPENAI_API_KEY in environment for LLM scorers.
- weave is optional; if unavailable, decorators no-op.
"""

# ------------- MUST set env BEFORE importing numpy/opencv -------------
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_os.environ.setdefault("FLAGS_use_mkldnn", "0")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("KMP_AFFINITY", "disabled")
_os.environ.setdefault("KMP_BLOCKTIME", "0")
_os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
_os.environ.setdefault("FLAGS_cpu_deterministic", "1")
# ----------------------------------------------------------------------

import os
import io
import json
import argparse
import asyncio
import base64 as b64
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass
from PIL import Image
import pydantic
import openai

# Stage IV judge prompts live in prompts/evaluation.py (verbatim, supplement-consistent).
from prompts.evaluation import QUALITY_PROMPT, CONTENT_PROMPT, STYLE_PROMPT

# ---------------- Optional deps handling ----------------

try:
    import weave
except ImportError:  # graceful fallback
    class weave:
        class Scorer:
            pass
        @staticmethod
        def op(*_args, **_kwargs):
            def _wrap(f):
                return f
            return _wrap
else:
    # weave is used only as a thin tracing decorator here; we never call
    # weave.init(), so silence its "traces will not be logged" warnings.
    import logging as _logging
    for _name in ("weave", "weave.trace", "weave_trace", "wandb"):
        _logging.getLogger(_name).setLevel(_logging.ERROR)

# tqdm is optional; dataset-level progress is shown when available
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# ---------------- BaseScorer ----------------

class BaseScorer(weave.Scorer):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @weave.op()
    async def score(self,
                    style_reference: Image.Image,
                    content_input: Image.Image,
                    generated_output: Image.Image,
                    metadata: Dict[str, Any] = None) -> float:
        raise NotImplementedError

    @weave.op()
    async def batch_score(self,
                          style_references: List[Image.Image],
                          content_inputs: List[Image.Image],
                          generated_outputs: List[Image.Image],
                          metadata: List[Dict[str, Any]] = None) -> List[float]:
        tasks = []
        for i, (s, c, g) in enumerate(zip(style_references, content_inputs, generated_outputs)):
            meta = metadata[i] if metadata else {}
            tasks.append(self.score(s, c, g, meta))
        return await asyncio.gather(*tasks)


# ---------------- LLM schema ----------------

class ReasoningScoreOutput(pydantic.BaseModel):
    reasoning: str
    score: float


# ---------------- Helpers ----------------

def _pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return b64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------- LLMVisualQualityJudgeScorer ----------------

class LLMVisualQualityJudgeScorer(BaseScorer):
    """
    LLM-based perceptual quality (readability + technical quality) score, 1..5 (higher is better).

    Image-only evaluation, explicitly targeted to:
      - charts, and
      - FDT = flowcharts, diagrams, tables.
    The model is NOT asked to classify the image.
    Strong emphasis on detecting CROPPING / clipping at canvas edges.
    Focus on legibility, contrast, alignment, and rendering artifacts.
    """
    client: openai.AsyncOpenAI = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=90.0, max_retries=2)

    quality_prompt: str = QUALITY_PROMPT

    def __init__(self):
        super().__init__("llm_visual_quality_charts_and_fdt")

    def convert_pil_to_base64(self, image: Image.Image) -> str:
        bytes_io = io.BytesIO()
        image.save(bytes_io, format="PNG")
        bytes_io.seek(0)
        return b64.b64encode(bytes_io.getvalue()).decode("utf-8")

    @weave.op()
    async def _get_quality_score(self, generated_output: Image.Image) -> dict:
        gen_b64 = self.convert_pil_to_base64(generated_output)

        response = await self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.quality_prompt},
                        {"type": "text", "text": "Image to rate (charts & FDT):"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gen_b64}"}}
                    ]
                }
            ],
            response_format=ReasoningScoreOutput
        )
        out = response.choices[0].message
        if hasattr(out, "parsed") and isinstance(out.parsed, ReasoningScoreOutput):
            return out.parsed.model_dump()

        # Fallback parse path
        try:
            data = json.loads(getattr(out, "content", "") or "{}")
            return {"reasoning": data.get("reasoning", ""), "score": float(data.get("score", 3))}
        except Exception:
            return {"reasoning": getattr(out, "content", "") or "", "score": 3.0}

    @weave.op()
    async def score(self, style_reference: Image.Image, content_input: Image.Image,
                    generated_output: Image.Image, metadata: Dict[str, Any] = None) -> float:
        result = await self._get_quality_score(generated_output)
        return float(result["score"])


# ---------------- LLMContentJudgeScorer (STRICT character-level preservation) ----------------

class LLMContentJudgeScorer(BaseScorer):
    """
    LLM-based content similarity 1..5 (higher is more similar).
    STRICT: character-level preservation for charts and FDT (FDT = flowcharts, diagrams, tables).
    """
    client: openai.AsyncOpenAI = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=90.0, max_retries=2)
    model: str = "gpt-4o"
    content_prompt: str = CONTENT_PROMPT

    def __init__(self):
        super().__init__("llm_content_judge_strict_char_level_charts_and_fdt")

    def convert_pil_to_base64(self, image: Image.Image) -> str:
        bytes_io = io.BytesIO()
        image.save(bytes_io, format="PNG")
        bytes_io.seek(0)
        return b64.b64encode(bytes_io.getvalue()).decode("utf-8")

    @weave.op()
    async def _get_similarity_score(self, content_input: Image.Image, generated_output: Image.Image) -> dict:
        content_base64 = self.convert_pil_to_base64(content_input)
        generated_base64 = self.convert_pil_to_base64(generated_output)

        response = await self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.content_prompt},
                        {"type": "text", "text": "Content Input:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{content_base64}" }},
                        {"type": "text", "text": "Generated Output:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{generated_base64}" }}
                    ]
                }
            ],
            response_format=ReasoningScoreOutput
        )

        output_message = response.choices[0].message
        if hasattr(output_message, "parsed") and isinstance(output_message.parsed, ReasoningScoreOutput):
            return output_message.parsed.model_dump()

        try:
            data = json.loads(getattr(output_message, "content", "") or "{}")
            return {"reasoning": data.get("reasoning", ""), "score": float(data.get("score", 3))}
        except Exception:
            return {"reasoning": getattr(output_message, "content", "") or "", "score": 3.0}

    async def score(
        self, 
        style_reference: Image.Image,
        content_input: Image.Image, 
        generated_output: Image.Image,
        metadata: Dict[str, Any]
    ) -> float:
        score_dict = await self._get_similarity_score(content_input, generated_output)
        return float(score_dict["score"])


# ---------------- LLMStyleJudgeScorer (optimized style prompt) ----------------

class LLMStyleJudgeScorer(BaseScorer):
    """
    LLM-based style similarity 1..5 (higher is more similar) for charts and FDT (FDT = flowcharts, diagrams, tables),
    ignoring semantic/textual content identity and focusing purely on visual styling.
    """
    client: openai.AsyncOpenAI = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=90.0, max_retries=2)
    model: str = "gpt-4o"
    style_prompt: str = STYLE_PROMPT

    def __init__(self):
        super().__init__("llm_style_judge_optimized_charts_and_fdt")

    def convert_pil_to_base64(self, image: Image.Image) -> str:
        bytes_io = io.BytesIO()
        image.save(bytes_io, format="PNG")
        bytes_io.seek(0)
        return b64.b64encode(bytes_io.getvalue()).decode("utf-8")

    @weave.op()
    async def _get_similarity_score(self, style_reference: Image.Image, generated_output: Image.Image) -> dict:
        style_base64 = self.convert_pil_to_base64(style_reference)
        generated_base64 = self.convert_pil_to_base64(generated_output)

        response = await self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.style_prompt},
                        {"type": "text", "text": "Style Reference:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{style_base64}"}},
                        {"type": "text", "text": "Generated Output:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{generated_base64}"}}
                    ]
                }
            ],
            response_format=ReasoningScoreOutput
        )
        output_message = response.choices[0].message
        if hasattr(output_message, "parsed") and isinstance(output_message.parsed, ReasoningScoreOutput):
            return output_message.parsed.model_dump()

        try:
            data = json.loads(getattr(output_message, "content", "") or "{}")
            return {"reasoning": data.get("reasoning", ""), "score": float(data.get("score", 3))}
        except Exception:
            return {"reasoning": getattr(output_message, "content", "") or "", "score": 3.0}

    async def score(
        self, 
        style_reference: Image.Image,
        content_input: Image.Image, 
        generated_output: Image.Image,
        metadata: Dict[str, Any]
    ) -> float:
        score_dict = await self._get_similarity_score(style_reference, generated_output)
        return float(score_dict["score"])


# ---------------- Single-sample evaluation (no per-item prints) ----------------

@dataclass
class SingleResult:
    llm_visual_quality_content_1to5: float
    llm_visual_quality_target_1to5: float
    llm_content_similarity_1to5: float
    llm_style_similarity_1to5: float
    llm_visual_quality_content_reasoning: Optional[str] = None
    llm_visual_quality_target_reasoning: Optional[str] = None
    llm_content_reasoning: Optional[str] = None
    llm_style_reasoning: Optional[str] = None


async def evaluate_scores(style_reference: Image.Image,
                          content_input: Image.Image,
                          generated_output: Image.Image) -> Dict[str, float]:
    """
    Asynchronously compute four metrics for one triplet and return a dict.
    (No per-item/thread printing)
    """
    # 1) LLM Visual Quality on content image
    vq = LLMVisualQualityJudgeScorer()
    vq_dict_c = await vq._get_quality_score(content_input)
    vq_c = float(vq_dict_c["score"])
    vq_c_reason = vq_dict_c.get("reasoning", None)

    # 2) LLM Visual Quality on target image
    vq_dict_t = await vq._get_quality_score(generated_output)
    vq_t = float(vq_dict_t["score"])
    vq_t_reason = vq_dict_t.get("reasoning", None)

    # 3) LLM Content (STRICT character-level)
    llm_content = LLMContentJudgeScorer()
    content_dict = await llm_content._get_similarity_score(content_input, generated_output)
    content_score = float(content_dict["score"])
    content_reason = content_dict.get("reasoning", None)

    # 4) LLM Style
    llm_style = LLMStyleJudgeScorer()
    style_dict = await llm_style._get_similarity_score(style_reference, generated_output)
    style_score = float(style_dict["score"])
    style_reason = style_dict.get("reasoning", None)

    return {
        "llm_visual_quality_content_1to5": vq_c,
        "llm_visual_quality_target_1to5": vq_t,
        "llm_content_similarity_1to5": float(content_score),
        "llm_style_similarity_1to5": float(style_score),
        "llm_visual_quality_content_reasoning": vq_c_reason,
        "llm_visual_quality_target_reasoning": vq_t_reason,
        "llm_content_reasoning": content_reason,
        "llm_style_reasoning": style_reason,
    }


def evaluate_scores_sync(style_reference: Image.Image,
                         content_input: Image.Image,
                         generated_output: Image.Image) -> Dict[str, float]:
    """Synchronous wrapper for evaluate_scores(...)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(
            evaluate_scores(style_reference, content_input, generated_output),
            loop
        ).result()
    else:
        return asyncio.run(
            evaluate_scores(style_reference, content_input, generated_output)
        )


# ---------------- Batch evaluation + JSON I/O with dataset-level progress ----------------

BASE_DIR = "output/cvpr_chart_restyle"

def _ensure_dirs(ratings_dir: str) -> None:
    os.makedirs(ratings_dir, exist_ok=True)

def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _avg(vals: List[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))

def _safe_open_image(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA", "LA", "L"):
        im = im.convert("RGB")
    return im

async def _evaluate_one_async(
    sem: "asyncio.Semaphore",
    sample_id: str,
    style_path: str,
    content_path: str,
    target_path: str,
    timeout_s: float,
) -> Dict[str, Any]:
    """Score a single triplet inside the shared event loop, bounded by a timeout.

    A failed/timed-out item is recorded with neutral (3.0) scores and an ``error``
    field rather than hanging or crashing the whole batch.
    """
    async with sem:
        try:
            style_im = _safe_open_image(style_path)
            content_im = _safe_open_image(content_path)
            target_im = _safe_open_image(target_path)
            result = await asyncio.wait_for(
                evaluate_scores(style_im, content_im, target_im), timeout=timeout_s
            )
        except Exception as e:  # timeout, API error, decode error, ...
            result = {
                "llm_visual_quality_content_1to5": 3.0,
                "llm_visual_quality_target_1to5": 3.0,
                "llm_content_similarity_1to5": 3.0,
                "llm_style_similarity_1to5": 3.0,
                "error": f"{type(e).__name__}: {e}",
            }
    return {
        "sample_id": sample_id,
        "style_path": style_path,
        "content_path": content_path,
        "target_path": target_path,
        **result,
    }


async def _evaluate_batch_async(
    items: List[Tuple[str, str, str, str]],
    concurrency: int,
    timeout_s: float,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))
    prog = tqdm(total=len(items), desc="Stage IV-a quality", unit="item") if (tqdm and show_progress) else None
    tasks = [
        asyncio.ensure_future(_evaluate_one_async(sem, sid, sp, cp, tp, timeout_s))
        for (sid, sp, cp, tp) in items
    ]
    collected: List[Dict[str, Any]] = []
    for fut in asyncio.as_completed(tasks):
        collected.append(await fut)
        if prog:
            prog.update(1)
    if prog:
        prog.close()
    return collected


def evaluate_and_save_from_lists_sync(
    items: List[Tuple[str, str, str, str]],    # (sample_id, style, content, target)
    *,
    ratings_dir: str,
    num_workers: int = 4,
    show_progress: bool = True,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """Single-event-loop batch evaluation from an explicit triplets list.

    All triplets are scored concurrently within ONE asyncio event loop (bounded by
    a semaphore), so the shared AsyncOpenAI clients are never used across loops.
    """
    _ensure_dirs(ratings_dir)

    # LLM judging is I/O bound, so we allow high concurrency (e.g. --num-workers 96).
    # A generous ceiling only guards against pathological values.
    concurrency = max(1, min(num_workers, 256))
    all_items = asyncio.run(
        _evaluate_batch_async(items, concurrency, timeout_s, show_progress)
    )

    vq_c_vals = [float(d["llm_visual_quality_content_1to5"]) for d in all_items]
    vq_t_vals = [float(d["llm_visual_quality_target_1to5"]) for d in all_items]
    content_vals = [float(d["llm_content_similarity_1to5"]) for d in all_items]
    style_vals = [float(d["llm_style_similarity_1to5"]) for d in all_items]
    n_errors = sum(1 for d in all_items if d.get("error"))

    all_items.sort(key=lambda d: d.get("sample_id", ""))
    summary = {
        "num_items": len(items),
        "num_failed": n_errors,
        "llm_visual_quality_content_1to5_avg": _avg(vq_c_vals),
        "llm_visual_quality_target_1to5_avg": _avg(vq_t_vals),
        "llm_content_similarity_1to5_avg": _avg(content_vals),
        "llm_style_similarity_1to5_avg": _avg(style_vals),
    }

    # Write ONE consolidated scores file (summary + every item's scores).
    out_path = os.path.join(ratings_dir, "quality_scores.json")
    _write_json(out_path, {"summary": summary, "items": all_items})
    summary["scores_path"] = out_path
    return summary


# ---------------- CLI: jsonl or single triple ----------------

def _load_jsonl_items(jsonl_path: str, max_items: Optional[int] = None) -> List[Tuple[str, str, str, str]]:
    """Return list of (sample_id, style_path, content_path, target_path). Ignores rows with missing/unreadable paths."""
    items: List[Tuple[str, str, str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_items is not None and len(items) >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            style_p = obj.get("style_image")
            content_p = obj.get("content_image")
            target_p = obj.get("target_image")
            if not (style_p and content_p and target_p):
                continue
            if not (os.path.exists(style_p) and os.path.exists(content_p) and os.path.exists(target_p)):
                continue
            sid = obj.get("id") or obj.get("sample_id") or f"pair_{idx:06d}"
            items.append((sid, style_p, content_p, target_p))
    return items


def main():
    parser = argparse.ArgumentParser(description="LLM-only evaluation for charts and FDT (FDT = flowcharts, diagrams, tables): Visual Quality + Content (strict char-level) + Style")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--jsonl", type=str, help="Path to local JSONL with fields: style_image, content_image, target_image")
    mode.add_argument("--style", type=str, help="Path to style reference image (single run)")

    # If single mode:
    parser.add_argument("--content", type=str, help="Path to content image (single run)")
    parser.add_argument("--target", type=str, help="Path to target/generated image (single run)")

    parser.add_argument("--ratings-dir", type=str, default="output/cvpr_chart_restyle/ratings",
                        help="Directory to write per-item JSON ratings (default: output/cvpr_chart_restyle/ratings)")
    parser.add_argument("--num-workers", type=int, default=4, help="Thread pool size for JSONL batch")
    parser.add_argument("--max-items", type=int, default=None, help="Max items to read from JSONL")

    parser.add_argument("--no-progress", action="store_true", help="Disable dataset-level progress bar")

    args = parser.parse_args()

    show_progress = not args.no_progress

    if args.jsonl:
        # Batch mode from JSONL (main-process-only progress)
        items = _load_jsonl_items(args.jsonl, max_items=args.max_items)
        if not items:
            print(f"[ERROR] No valid items found in JSONL: {args.jsonl}")
            raise SystemExit(2)

        overall = evaluate_and_save_from_lists_sync(
            items,
            ratings_dir=args.ratings_dir,
            num_workers=args.num_workers,
            show_progress=show_progress,
        )
        print(json.dumps(overall, ensure_ascii=False, indent=2))
        print(f"Quality scores written to: {overall['scores_path']}")
        return

    # Single mode (no per-item progress)
    if not (args.style and args.content and args.target):
        print("[ERROR] For single run, please provide --style, --content, and --target.")
        raise SystemExit(2)
    for pth in (args.style, args.content, args.target):
        if not os.path.exists(pth):
            print(f"[ERROR] File does not exist: {pth}")
            raise SystemExit(2)

    style_im = _safe_open_image(args.style)
    content_im = _safe_open_image(args.content)
    target_im = _safe_open_image(args.target)

    result = evaluate_scores_sync(style_im, content_im, target_im)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Also write a one-off JSON
    os.makedirs(args.ratings_dir, exist_ok=True)
    out_path = os.path.join(args.ratings_dir, "single_item.json")
    _write_json(out_path, result)
    print(f"Single result written to: {out_path}")


if __name__ == "__main__":
    main()

"""
Examples:

# Very large run (main-process progress only)
python stage4a_quality_filtering.py \
  --jsonl output/cvpr_chart_restyle/training_set_300k_all_for_chart_style_transfer.jsonl \
  --ratings-dir output/cvpr_chart_restyle/ratings \
  --num-workers 96 \
  --max-items 500000
"""
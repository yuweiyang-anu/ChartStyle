#!/usr/bin/env python3
"""ChartStyle-Bench evaluation entry point.

Generates restyled images for the ChartStyle-Bench benchmark with a chosen model
and scores them locally with CLIP, LLM-as-judge, and OCR metrics. Everything runs
on the local machine and results are written to ``--output-dir`` as JSON + PNGs.

Examples
--------
    # OpenAI gpt-image-1 (paper setting) on the first 20 pairs, 8-way concurrency:
    export OPENAI_API_KEY=sk-...
    python run_eval.py --model gpt-image-1 --limit 20 --concurrency 8

    # Local Qwen-Image-Edit base model (8-step fast path), all visible GPUs:
    python run_eval.py --model qwen-image-edit --qwen-fast

    # Nano Banana Pro (Gemini) with a subset of metrics:
    export GOOGLE_API_KEY=...
    python run_eval.py --model nano-banana-pro --scorers content style leakage

Run from inside the ``evaluation/`` directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv

from dataset import load_bench
from harness import EvaluationHarness

load_dotenv()


# --- Inference models -------------------------------------------------------
# Factories import their (heavy) dependencies lazily so a missing backend dep
# (e.g. diffusers for Qwen) does not break the others.
def _build_qwen(args):
    from models.qwen_image_edit import MultiGPUQwenImageEditModel
    return MultiGPUQwenImageEditModel(
        model_id=args.qwen_model_id,
        max_gpus=args.max_gpus,
        fast=args.qwen_fast,
        num_inference_steps=args.qwen_steps,
    )


def _build_openai(model_name):
    def factory(args):
        from models.openai_image import OpenAIStyleTransferModel
        return OpenAIStyleTransferModel(model_name=model_name, api_key=args.openai_api_key)
    return factory


def _build_nano_banana(args):
    from models.nano_banana import NanoBananaProModel
    return NanoBananaProModel(model_name=args.gemini_model, api_key=args.google_api_key)


MODEL_REGISTRY = {
    "qwen-image-edit": _build_qwen,
    "gpt-image-1": _build_openai("gpt-image-1"),
    "gpt-image-1.5": _build_openai("gpt-image-1.5"),
    "gpt-image-2": _build_openai("gpt-image-2"),
    "nano-banana-pro": _build_nano_banana,
}


# --- Scorers (CLI aliases; result keys use the paper's full metric names) ---
def _scorer_factories(args):
    # Non-LLM metrics (CLIP / OCR).
    def fidelity():
        from scorers.clip_scorers import CLIPStylisticFidelityScorer
        return CLIPStylisticFidelityScorer()

    def semantic():
        from scorers.clip_scorers import SemanticConsistencyScorer
        return SemanticConsistencyScorer()

    def ocrscore():
        from scorers.ocr_scorer import PaddleOCROverlapScorer
        return PaddleOCROverlapScorer()

    # LLM Evaluation metrics (GPT-4o judges); Overall Score is derived in the harness.
    def content():
        from scorers.llm_judges import LLMContentJudgeScorer
        return LLMContentJudgeScorer(model=args.judge_model, api_key=args.openai_api_key)

    def style():
        from scorers.llm_judges import LLMStyleJudgeScorer
        return LLMStyleJudgeScorer(model=args.judge_model, api_key=args.openai_api_key)

    def leakage():
        from scorers.llm_judges import LLMContentLeakageScorer
        return LLMContentLeakageScorer(model=args.judge_model, api_key=args.openai_api_key)

    return {
        "content": content,
        "style": style,
        "leakage": leakage,
        "semantic": semantic,
        "fidelity": fidelity,
        "ocrscore": ocrscore,
    }


# All six benchmark metrics by default (Overall Score is derived automatically).
DEFAULT_SCORERS = ["content", "style", "leakage", "semantic", "fidelity", "ocrscore"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ChartStyle-Bench evaluation locally.")
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY), help="Inference model to evaluate.")
    p.add_argument("--scorers", nargs="*", default=DEFAULT_SCORERS,
                   choices=list(_scorer_factories(argparse.Namespace(judge_model=None, openai_api_key=None))),
                   help="Subset of scorers to run (default: all).")
    p.add_argument("--dataset-repo", default="ChartFoundation/ChartStyleBench", help="HF dataset repo id.")
    p.add_argument("--split", default="test", help="Dataset split.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples.")
    p.add_argument("--concurrency", type=int, default=4, help="Max concurrent generate+score pipelines.")
    p.add_argument("--output-dir", default="eval_results", help="Where to write results.")
    p.add_argument("--no-images", action="store_true", help="Do not save generated images to disk.")
    p.add_argument("--judge-model", default="gpt-4o", help="OpenAI model for the LLM judges.")
    # Credentials (fall back to environment variables).
    p.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--google-api-key", default=os.getenv("GOOGLE_API_KEY"))
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HF token for private/gated datasets.")
    # Model-specific.
    p.add_argument("--qwen-model-id", default=None, help="Override the Qwen base-model id/path.")
    p.add_argument("--max-gpus", type=int, default=None, help="Cap GPUs for the multi-GPU Qwen model.")
    p.add_argument("--qwen-fast", action="store_true",
                   help="Use the 8-step Lightning LoRA path for Qwen (default: 40-step paper setting).")
    p.add_argument("--qwen-steps", type=int, default=None,
                   help="Explicit Qwen denoising-step override (e.g. 40 or 8).")
    p.add_argument("--gemini-model", default="gemini-3-pro-image-preview", help="Gemini image model id.")
    return p.parse_args()


async def _main_async(args, model, scorers):
    print(f"Loading {args.dataset_repo} (split={args.split}, limit={args.limit})...")
    samples = load_bench(args.dataset_repo, split=args.split, limit=args.limit, token=args.hf_token)
    print(f"Loaded {len(samples)} samples.")

    harness = EvaluationHarness(
        model=model,
        scorers=scorers,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        save_images=not args.no_images,
    )
    summary = await harness.run(samples)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nFull results: {harness.run_dir}")


def main() -> None:
    args = parse_args()
    print(f"Building model: {args.model}")
    model = MODEL_REGISTRY[args.model](args)

    factories = _scorer_factories(args)
    print(f"Building scorers: {', '.join(args.scorers)}")
    scorers = [factories[name]() for name in args.scorers]

    asyncio.run(_main_async(args, model, scorers))


if __name__ == "__main__":
    main()

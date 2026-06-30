# -*- coding: utf-8 -*-
"""
ChartForge — Stage IV, step 3 of 3 (FINAL): threshold aggregation and selection.

Last step of Stage IV (and of the whole pipeline). It joins the two consolidated
score files produced earlier in Stage IV, on their shared ``sample_id``:

  - quality_scores.json  (stage4a) : four LLM judgements per triplet, each 1-5.
  - ocr_scores.json      (stage4b) : PaddleOCR text-overlap F1 per triplet, 0-1.

A per-metric threshold is applied to every dimension; triplets that clear *all*
thresholds are written to a JSONL file (style / content / target image paths) for
downstream packaging into ChartStyle-100K.

Dimensions and default thresholds (defaults match the paper's main filter, i.e.
the "4-4-3-3-0.8" configuration):

  - quality_content (>= 4)    visual quality of the content image  I_c
  - quality_target  (>= 4)    visual quality of the target image   I_t
  - content_match   (>= 3)    content preservation (I_c vs I_t)
  - style_match     (>= 3)    style transfer       (I_s vs I_t)
  - ocr_f1          (>= 0.8)  OCR text-overlap F1  (I_c vs I_t)

Example:
    python stage4c_aggregate_filter.py \
        --quality-json output/stage4/ratings_quality/quality_scores.json \
        --ocr-json     output/stage4/ratings_ocr/ocr_scores.json \
        --out-jsonl    output/stage4/filtered_triplets.jsonl
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(float(x))


def _load_items(path: str) -> List[Dict[str, Any]]:
    """Load a consolidated scores file.

    Accepts either the ``{"summary": ..., "items": [...]}`` layout written by
    Stage IV-a/IV-b, or a bare ``[...]`` list of item dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    if isinstance(blob, dict) and "items" in blob:
        return blob["items"]
    if isinstance(blob, list):
        return blob
    raise ValueError(f"Unrecognized scores-file structure (expected dict with 'items' or a list): {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ChartForge Stage IV-c: join LLM-quality + OCR scores, threshold, export training JSONL."
    )
    ap.add_argument("--quality-json", required=True, help="stage4a output: quality_scores.json")
    ap.add_argument("--ocr-json", required=True, help="stage4b output: ocr_scores.json")
    ap.add_argument("--out-jsonl", required=True, help="output filtered triplets (JSONL)")
    ap.add_argument("--min-quality-content", type=float, default=4.0)
    ap.add_argument("--min-quality-target", type=float, default=4.0)
    ap.add_argument("--min-content-match", type=float, default=3.0)
    ap.add_argument("--min-style-match", type=float, default=3.0)
    ap.add_argument("--min-ocr-f1", type=float, default=0.8)
    args = ap.parse_args()

    quality = _load_items(args.quality_json)
    ocr = _load_items(args.ocr_json)
    ocr_by_id = {o.get("sample_id"): o for o in ocr}

    thresholds = {
        "quality_content": args.min_quality_content,
        "quality_target": args.min_quality_target,
        "content_match": args.min_content_match,
        "style_match": args.min_style_match,
        "ocr_f1": args.min_ocr_f1,
    }

    # Join the two score sources on sample_id into a flat per-triplet record.
    joined: List[Dict[str, Any]] = []
    for q in quality:
        sid = q.get("sample_id")
        o = ocr_by_id.get(sid)
        joined.append({
            "sample_id": sid,
            "style_image": q.get("style_path"),
            "content_image": q.get("content_path"),
            "target_image": q.get("target_path"),
            "quality_content": q.get("llm_visual_quality_content_1to5"),
            "quality_target": q.get("llm_visual_quality_target_1to5"),
            "content_match": q.get("llm_content_similarity_1to5"),
            "style_match": q.get("llm_style_similarity_1to5"),
            "ocr_f1": (o.get("paddleocr_text_overlap_f1") if o else None),
        })

    print(f"Quality items: {len(quality)} | OCR items: {len(ocr)} | joined: {len(joined)}")
    print("=== Per-metric statistics ===")
    for m in ("quality_content", "quality_target", "content_match", "style_match", "ocr_f1"):
        vals = [r[m] for r in joined if _is_number(r[m])]
        mean = (sum(vals) / len(vals)) if vals else float("nan")
        n_pass = sum(1 for v in vals if v >= thresholds[m])
        print(f"  {m:16s} mean={mean:.4f}  >= {thresholds[m]}: {n_pass}/{len(vals)}")

    def passes(r: Dict[str, Any]) -> bool:
        for metric, thr in thresholds.items():
            v = r[metric]
            if not _is_number(v) or float(v) < float(thr):
                return False
        return True

    out_dir = os.path.dirname(os.path.abspath(args.out_jsonl)) or "."
    os.makedirs(out_dir, exist_ok=True)

    kept = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in joined:
            if not passes(r):
                continue
            if not (r["style_image"] and r["content_image"] and r["target_image"]):
                continue
            f.write(json.dumps({
                "style_image": r["style_image"],
                "content_image": r["content_image"],
                "target_image": r["target_image"],
            }, ensure_ascii=False) + "\n")
            kept += 1

    print("=== Selection ===")
    print(f"Kept {kept}/{len(joined)} triplets clearing all thresholds -> {args.out_jsonl}")


if __name__ == "__main__":
    main()

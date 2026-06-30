# -*- coding: utf-8 -*-
"""
ChartForge — Stage I: Reference-driven target generation.

Given a style exemplar ``I_s`` alone, synthesize a stylized **target** ``I_t``
whose visual appearance follows the reference while its underlying content
(visualization type, subject, layout, density) is sampled independently from the
curated pools. This is the *primary* generation step; the structurally aligned
content image ``I_c`` is produced later in Stage II.

Two domains are supported via ``--domain``:

- ``chart`` : fine-grained charts (36 chart types).
- ``fdt``   : flowcharts / diagrams / tables, discovered from the ``canva-charts``
              and ``chart-galaxy-real`` reference sub-folders.

Each generated primary is logged to ``<out>/json/primary__*.json`` and collected
into a run manifest. Stage II (``stage2_content_generation.py``) consumes those
primary JSONs, so this stage is naturally resumable: re-running only regenerates
missing primaries if you point it at a fresh window.

Example (charts):
    python stage1_target_generation.py --domain chart \
        --images-subdir cvpr_chart_restyle/overall_dataset_v1/reference_images/chart-galaxy-real \
        --out-subdir   cvpr_chart_restyle/overall_dataset_v1/stylized_chart_galaxy_real_images \
        --image-start 40000 --image-end 59999 \
        --concurrency 96 --model gpt-image-1 \
        --manifest manifest_chart_galaxy_40000_59999_primary.json

Example (FDT):
    python stage1_target_generation.py --domain fdt \
        --images-subdir cvpr_chart_restyle/overall_dataset_v1_stage2/reference_images \
        --out-subdir   cvpr_chart_restyle/overall_dataset_v1_stage2 \
        --image-start 0 --image-end 20000 \
        --concurrency 96 --model gpt-image-1 \
        --manifest manifest_mix_flow_diagram_table_unified.json
"""

import argparse
import json
import os
import random
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

from openai import OpenAI

from common import (
    b64_to_png,
    call_images_edit_with_file,
    discover_style_images,
    iter_images_recursive,
    safe_slug,
    shorten_stem,
)
from configs import pools
from prompts import chart_generation as chart_prompts
from prompts import fdt_generation as fdt_prompts


# All inputs and outputs live under one local base directory. Defaults to
# "./output" (relative to the current working directory); override with the
# CHARTFORGE_OUTPUT_ROOT environment variable. No remote/object storage is used.
ROOT = Path(os.environ.get("CHARTFORGE_OUTPUT_ROOT", "output"))


# ================================================================
# CHART DOMAIN
# ================================================================

@dataclass
class ChartPrimarySpec:
    style_path: Path
    model: str
    chart_type: str
    subject: str
    layout_type: str           # "single" or "freeform"
    num_panels: int
    info_density: str
    out_dir_primary: Path
    json_dir: Path
    run_id: str
    panel_chart_types: List[str]
    complexity_from_reference: bool
    subject_driven_titles: bool


def _choose_layout(single_prob: float) -> Tuple[str, int]:
    """Randomly choose between single-panel or multi-panel layout."""
    if random.random() < single_prob:
        return "single", 1
    return "freeform", random.randint(2, 12)


def _choose_panel_chart_types(num_panels: int) -> List[str]:
    if num_panels == 1:
        return [random.choice(pools.CHART_TYPES)]
    k = random.choice([1, 2, 3])
    return random.sample(pools.CHART_TYPES, k)


def make_chart_primary_specs(style_path: Path, model: str, per_style_primary: int,
                             out_dir_primary: Path, json_dir: Path, run_id: str,
                             single_prob: float, complexity_from_reference: bool,
                             subject_driven_titles: bool) -> List[ChartPrimarySpec]:
    specs: List[ChartPrimarySpec] = []
    for _ in range(per_style_primary):
        chart_type = random.choice(pools.CHART_TYPES)
        subject = random.choice(pools.SUBJECTS)
        layout_type, num_panels = _choose_layout(single_prob)
        info_density = random.choice(pools.INFO_DENSITIES)
        panel_types = _choose_panel_chart_types(num_panels)
        specs.append(
            ChartPrimarySpec(
                style_path=style_path, model=model, chart_type=chart_type, subject=subject,
                layout_type=layout_type, num_panels=num_panels, info_density=info_density,
                out_dir_primary=out_dir_primary, json_dir=json_dir, run_id=run_id,
                panel_chart_types=panel_types, complexity_from_reference=complexity_from_reference,
                subject_driven_titles=subject_driven_titles,
            )
        )
    return specs


def generate_chart_primary(client: OpenAI, spec: ChartPrimarySpec) -> Dict:
    prompt = chart_prompts.build_primary_prompt(
        spec.chart_type, spec.subject, spec.layout_type, spec.num_panels,
        spec.style_path.name, spec.info_density, spec.panel_chart_types,
        spec.complexity_from_reference, spec.subject_driven_titles,
    )

    result = call_images_edit_with_file(client, spec.model, spec.style_path, prompt)
    if not hasattr(result, "data") or not result.data or not getattr(result.data[0], "b64_json", None):
        raise RuntimeError("images.edit returned no b64_json payload for primary")

    ts_ms = int(time.time() * 1000)
    style_stem = spec.style_path.stem
    layout_tag = "single" if spec.layout_type == "single" else "multi"
    safe_chart = spec.chart_type.replace(" ", "_").replace("/", "_")
    safe_subj = spec.subject.replace(" ", "_")
    out_path = spec.out_dir_primary / f"{style_stem}__A_primary__{safe_chart}__{safe_subj}__{layout_tag}__{ts_ms}.png"
    b64_to_png(result.data[0].b64_json, out_path)

    record: Dict = {
        "stage": "primary",
        "run_id": spec.run_id,
        "timestamp_ms": ts_ms,
        "style_image": str(spec.style_path),
        "primary_output": str(out_path),
        "chart_type": spec.chart_type,
        "subject": spec.subject,
        "layout": {"layout_type": spec.layout_type, "num_panels": spec.num_panels},
        "panel_chart_types": spec.panel_chart_types,
        "model": spec.model,
        "info_density": spec.info_density,
        "complexity_from_reference": spec.complexity_from_reference,
        "subject_driven_titles": spec.subject_driven_titles,
        "prompt": prompt,
    }

    spec.json_dir.mkdir(parents=True, exist_ok=True)
    json_path = spec.json_dir / f"primary__{ts_ms}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def run_charts(args) -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    images_dir = ROOT / args.images_subdir
    # Stage I writes its targets under <out-subdir>/stage1_output/target.
    out_root = ROOT / args.out_subdir / "stage1_output"
    out_target = out_root / "target"
    json_dir = out_root / "json"
    for p in [out_root, out_target, json_dir]:
        p.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / args.manifest

    style_images = discover_style_images(images_dir, args.image_start, args.image_end)
    if not style_images:
        print(f"No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    run_id = f"run_{int(time.time())}_{random.randint(1000,9999)}"
    specs: List[ChartPrimarySpec] = []
    for style_path in style_images:
        specs.extend(
            make_chart_primary_specs(
                style_path=style_path, model=args.model, per_style_primary=args.per_style_primary,
                out_dir_primary=out_target, json_dir=json_dir, run_id=run_id,
                single_prob=args.single_panel_prob,
                complexity_from_reference=args.complexity_from_reference,
                subject_driven_titles=args.subject_driven_titles,
            )
        )

    primaries: List[Dict] = []
    errors: List[Tuple[Path, Exception]] = []
    lock = threading.Lock()
    pbar = tqdm(total=len(specs), desc="Stage I: chart primaries", unit="img") if tqdm else None
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(generate_chart_primary, client, spec): spec for spec in specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                rec = fut.result()
                with lock:
                    primaries.append(rec)
                    if pbar: pbar.update(1)
            except Exception as e:
                with lock:
                    errors.append((spec.style_path, e))
                    if pbar: pbar.update(1)
                print(f"[I][ERROR] {spec.style_path.name}: {e}", file=sys.stderr)
    if pbar: pbar.close()

    manifest = {
        "run": {
            "run_id": run_id, "timestamp_ms": int(time.time() * 1000),
            "domain": "chart", "stage": "target_generation",
            "images_dir": str(images_dir.resolve()),
            "output_dir_target": str(out_target.resolve()),
            "json_dir": str(json_dir.resolve()),
            "model": args.model, "per_style_primary": args.per_style_primary,
            "concurrency": args.concurrency, "single_panel_prob": args.single_panel_prob,
            "complexity_from_reference": args.complexity_from_reference,
            "subject_driven_titles": args.subject_driven_titles,
            "style_image_count": len(style_images),
            "primary_count": len(primaries), "errors_primary": len(errors),
        },
        "primaries": primaries,
        "errors_primary": [{"file": str(p), "error": str(e)} for p, e in errors],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n--- Stage I (charts) Summary ---")
    print(f"Primaries: {len(primaries)} / {len(specs)}")
    print(f"Manifest written to {manifest_path.resolve()}")


# ================================================================
# FDT DOMAIN (flowchart / diagram / table)
# ================================================================

@dataclass
class FDTPrimarySpec:
    style_path: Path
    model: str
    task_type: str             # "flowchart" | "diagram" | "table"
    subject: str
    layout_type: str           # "single" | "freeform"
    num_panels: int
    info_density: str
    out_dir_primary: Path
    json_dir: Path
    run_id: str
    complexity_from_reference: bool
    subject_driven_titles: bool
    source_bucket: str         # "canva" | "galaxy"


def make_fdt_primary_specs(style_path: Path, model: str, per_style_primary: int,
                           out_dir_primary: Path, json_dir: Path, run_id: str,
                           complexity_from_reference: bool, subject_driven_titles: bool,
                           single_prob: float, forced_task: Optional[str],
                           source_bucket: str) -> List[FDTPrimarySpec]:
    specs: List[FDTPrimarySpec] = []
    for _ in range(per_style_primary):
        subject = random.choice(pools.SUBJECTS)
        layout_type, num_panels = _choose_layout(single_prob)
        info_density = random.choice(pools.INFO_DENSITIES)
        if forced_task and forced_task != "auto":
            task = forced_task
        else:
            task = random.choice(pools.TASK_TYPES)
        specs.append(
            FDTPrimarySpec(
                style_path=style_path, model=model, task_type=task, subject=subject,
                layout_type=layout_type, num_panels=num_panels, info_density=info_density,
                out_dir_primary=out_dir_primary, json_dir=json_dir, run_id=run_id,
                complexity_from_reference=complexity_from_reference,
                subject_driven_titles=subject_driven_titles, source_bucket=source_bucket,
            )
        )
    return specs


def generate_fdt_primary(client: OpenAI, spec: FDTPrimarySpec) -> Dict:
    prompt = fdt_prompts.build_primary_prompt(
        task_type=spec.task_type, subject=spec.subject, layout_type=spec.layout_type,
        num_panels=spec.num_panels, info_density=spec.info_density,
        complexity_from_reference=spec.complexity_from_reference,
        subject_driven_titles=spec.subject_driven_titles,
    )

    result = call_images_edit_with_file(client, spec.model, spec.style_path, prompt)
    if not hasattr(result, "data") or not result.data or not getattr(result.data[0], "b64_json", None):
        raise RuntimeError("images.edit returned no b64_json payload for primary")

    ts_ms = int(time.time() * 1000)
    style_stem = safe_slug(spec.style_path.stem, max_len=48)
    layout_tag = "single" if spec.layout_type == "single" else "multi"
    safe_kind = safe_slug(spec.task_type, max_len=16)
    safe_subj = safe_slug(spec.subject, max_len=32)
    file_stem = f"{style_stem}__A_primary__{safe_kind}__{safe_subj}__{layout_tag}__{ts_ms}"
    file_stem = shorten_stem(file_stem, max_len=160)
    out_path = spec.out_dir_primary / f"{file_stem}.png"
    b64_to_png(result.data[0].b64_json, out_path)

    record: Dict = {
        "stage": "primary",
        "run_id": spec.run_id,
        "timestamp_ms": ts_ms,
        "style_image": str(spec.style_path.resolve()),
        "primary_output": str(out_path.resolve()),
        "subject": spec.subject,
        "layout": {"layout_type": spec.layout_type, "num_panels": spec.num_panels},
        "model": spec.model,
        "info_density": spec.info_density,
        "complexity_from_reference": spec.complexity_from_reference,
        "subject_driven_titles": spec.subject_driven_titles,
        "task_type": spec.task_type,
        "prompt": prompt,
        "source_bucket": spec.source_bucket,
    }

    spec.json_dir.mkdir(parents=True, exist_ok=True)
    json_path = spec.json_dir / f"primary__{ts_ms}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def run_fdt(args) -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    images_root = (ROOT / args.images_subdir).resolve()
    ref_canva = images_root / "canva-charts"
    ref_galaxy = images_root / "chart-galaxy-real"

    out_base = (ROOT / args.out_subdir / "stage1_output").resolve()
    out_canva = out_base / "stylized_canva_images"
    out_galaxy = out_base / "stylized_chart_galaxy_real_images"

    def ensure_set_dirs(out_set: Path) -> Tuple[Path, Path]:
        tdir = out_set / "target"
        jdir = out_set / "json"
        for p in [out_set, tdir, jdir]:
            p.mkdir(parents=True, exist_ok=True)
        return tdir.resolve(), jdir.resolve()

    canva_primary_dir, canva_json_dir = ensure_set_dirs(out_canva)
    galaxy_primary_dir, galaxy_json_dir = ensure_set_dirs(out_galaxy)
    manifest_path = (out_base / args.manifest).resolve()

    canva_refs = iter_images_recursive(ref_canva)
    galaxy_refs = iter_images_recursive(ref_galaxy)

    def slice_by_index(files: List[Path], start: int, end: Optional[int]) -> List[Path]:
        if not files:
            return []
        if end is None:
            return files[start:]
        return files[start:min(len(files), max(0, end + 1))]

    canva_refs = slice_by_index(canva_refs, args.image_start, args.image_end)
    galaxy_refs = slice_by_index(galaxy_refs, args.image_start, args.image_end)
    if not canva_refs and not galaxy_refs:
        print(f"No reference images found under {images_root}/(canva-charts|chart-galaxy-real)", file=sys.stderr)
        sys.exit(1)

    run_id = f"run_mix_{int(time.time())}_{random.randint(1000,9999)}"
    forced_task = args.task_type if args.task_type != "auto" else None

    specs: List[FDTPrimarySpec] = []
    for style_path in canva_refs:
        specs.extend(make_fdt_primary_specs(
            style_path, args.model, args.per_style_primary, canva_primary_dir, canva_json_dir,
            run_id, args.complexity_from_reference, args.subject_driven_titles,
            args.single_panel_prob, forced_task, "canva"))
    for style_path in galaxy_refs:
        specs.extend(make_fdt_primary_specs(
            style_path, args.model, args.per_style_primary, galaxy_primary_dir, galaxy_json_dir,
            run_id, args.complexity_from_reference, args.subject_driven_titles,
            args.single_panel_prob, forced_task, "galaxy"))

    primaries: List[Dict] = []
    errors: List[Tuple[Path, Exception]] = []
    lock = threading.Lock()
    pbar = tqdm(total=len(specs), desc="Stage I: FDT primaries", unit="img") if tqdm else None
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(generate_fdt_primary, client, spec): spec for spec in specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                rec = fut.result()
                with lock:
                    primaries.append(rec)
                    if pbar: pbar.update(1)
            except Exception as e:
                with lock:
                    errors.append((spec.style_path, e))
                    if pbar: pbar.update(1)
                print(f"[I][ERROR] {spec.style_path}: {e}", file=sys.stderr)
    if pbar: pbar.close()

    manifest = {
        "run": {
            "run_id": run_id, "timestamp_ms": int(time.time() * 1000),
            "domain": "fdt", "stage": "target_generation",
            "images_root": str(images_root),
            "reference_sets": {"canva": str(ref_canva), "galaxy": str(ref_galaxy)},
            "output_sets": {
                "canva": {"root": str(out_canva), "target": str(canva_primary_dir), "json": str(canva_json_dir)},
                "galaxy": {"root": str(out_galaxy), "target": str(galaxy_primary_dir), "json": str(galaxy_json_dir)},
            },
            "manifest_path": str(manifest_path),
            "model": args.model, "per_style_primary": args.per_style_primary,
            "concurrency": args.concurrency, "single_panel_prob": args.single_panel_prob,
            "complexity_from_reference": args.complexity_from_reference,
            "subject_driven_titles": args.subject_driven_titles,
            "task_type_mode": args.task_type,
            "primary_count": len(primaries), "errors_primary": len(errors),
        },
        "primaries": primaries,
        "errors_primary": [{"file": str(p), "error": str(e)} for p, e in errors],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n--- Stage I (FDT) Summary ---")
    print(f"Task type mode : {args.task_type}")
    print(f"Primaries      : {len(primaries)} / {len(specs)}")
    print(f"Manifest path  : {manifest_path}")


# ================================================================
# CLI
# ================================================================

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ChartForge Stage I: reference-driven target generation.")
    parser.add_argument("--domain", type=str, required=True, choices=["chart", "fdt"],
                        help="chart = fine-grained charts; fdt = flowchart/diagram/table.")
    parser.add_argument("--task-type", type=str, default="auto",
                        choices=["auto", "flowchart", "diagram", "table"],
                        help="(fdt only) fixed task type, or 'auto' to pick per primary.")
    parser.add_argument("--images-subdir", type=str, required=True)
    parser.add_argument("--out-subdir", type=str, required=True)
    parser.add_argument("--per-style-primary", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--model", type=str, default="gpt-image-1")
    parser.add_argument("--manifest", type=str, default="stage1_manifest.json")
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--image-end", type=int, default=None)
    parser.add_argument("--single-panel-prob", type=float, default=0.65,
                        help="Probability of sampling single-panel layout.")
    parser.add_argument("--complexity-from-reference", dest="complexity_from_reference",
                        action="store_true", default=True)
    parser.add_argument("--no-complexity-from-reference", dest="complexity_from_reference",
                        action="store_false")
    parser.add_argument("--subject-driven-titles", dest="subject_driven_titles",
                        action="store_true", default=True)
    parser.add_argument("--no-subject-driven-titles", dest="subject_driven_titles",
                        action="store_false")
    args = parser.parse_args()

    if not (0 <= args.single_panel_prob <= 1):
        print("ERROR: --single-panel-prob must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    if args.domain == "chart":
        run_charts(args)
    else:
        run_fdt(args)


if __name__ == "__main__":
    main()

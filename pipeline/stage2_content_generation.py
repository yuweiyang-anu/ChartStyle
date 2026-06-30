# -*- coding: utf-8 -*-
"""
ChartForge — Stage II: Restyle-based content generation.

With a stylized target ``I_t`` from Stage I, prompt the generator to reproduce
its content under a *different* visual family, yielding the structurally
identical content image ``I_c``. Because ``I_c`` is derived from ``I_t``, the
two stay structurally aligned and style is separated from content at the data
level. The output triplet ``(I_s, I_c, I_t)`` is the training unit.

This stage is **resume-friendly by construction**: it reads the per-primary JSON
logs written by Stage I (``<out>/json/primary__*.json``) rather than regenerating
anything. Re-running it simply restyles whatever primaries are present, so an
interrupted run can be continued by pointing it at the same output directory.

Two domains via ``--domain``:

- ``chart`` : single output tree; variant mix forces one minimal + one 3D family.
- ``fdt``   : dual ``canva`` / ``galaxy`` output trees; variant mix forces one
              minimal family (no forced 3D), rest sampled from the full pool.

Example (charts):
    python stage2_content_generation.py --domain chart \
        --out-subdir cvpr_chart_restyle/overall_dataset_v1/stylized_chart_galaxy_real_images \
        --variants-per 3 --concurrency 96 --model gpt-image-1 \
        --manifest manifest_chart_galaxy_40000_59999_restyled.json

Example (FDT):
    python stage2_content_generation.py --domain fdt \
        --out-subdir cvpr_chart_restyle/overall_dataset_v1_stage2 \
        --variants-per 2 --concurrency 96 --model gpt-image-1 \
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
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

from openai import OpenAI

from common import b64_to_png, call_images_edit_with_file, shorten_stem
from configs import pools
from configs.pools import FAMILY_ALIAS
from prompts import chart_generation as chart_prompts
from prompts import fdt_generation as fdt_prompts


# All inputs and outputs live under one local base directory. Defaults to
# "./output" (relative to the current working directory); override with the
# CHARTFORGE_OUTPUT_ROOT environment variable. No remote/object storage is used.
ROOT = Path(os.environ.get("CHARTFORGE_OUTPUT_ROOT", "output"))


def family_to_slug(family: str) -> str:
    from common import safe_slug
    return FAMILY_ALIAS.get(family, safe_slug(family, max_len=40))


# ================================================================
# CHART DOMAIN
# ================================================================

@dataclass
class ChartVariantSpec:
    base_image_path: Path
    model: str
    family_hint: str
    out_dir_variants: Path
    json_dir: Path
    run_id: str
    source_style_image: Path
    chart_type: str
    subject: str


def _chart_variant_mix(k: int) -> List[str]:
    """Charts policy: force one minimal family, try to include one 3D family."""
    if k <= 0:
        return []
    if k == 1:
        return [random.choice(pools.RESTYLE_FAMILIES)]
    minimal = random.choice(pools.REQUIRED_MINIMAL_POOL)
    remaining = [f for f in pools.RESTYLE_FAMILIES if f != minimal]
    rest_needed = k - 1
    selected = []
    if pools.THREED_FAMILIES and rest_needed > 0:
        threed_choice = random.choice(pools.THREED_FAMILIES)
        selected.append(threed_choice)
        remaining = [f for f in remaining if f != threed_choice]
        rest_needed -= 1
    if rest_needed > 0:
        selected.extend(random.sample(remaining, min(rest_needed, len(remaining))))
    return [minimal] + selected[:k - 1]


def make_chart_variant_specs(primary_record: Dict, variants_per: int, model: str,
                             out_dir_variants: Path, json_dir: Path) -> List[ChartVariantSpec]:
    base_image = Path(primary_record["primary_output"])
    k = max(1, min(variants_per, len(pools.RESTYLE_FAMILIES)))
    families = _chart_variant_mix(k)
    specs: List[ChartVariantSpec] = []
    for fam in families:
        specs.append(
            ChartVariantSpec(
                base_image_path=base_image, model=model, family_hint=fam,
                out_dir_variants=out_dir_variants, json_dir=json_dir,
                run_id=primary_record["run_id"],
                source_style_image=Path(primary_record["style_image"]),
                chart_type=primary_record["chart_type"], subject=primary_record["subject"],
            )
        )
    return specs


def restyle_chart_variant(client: OpenAI, spec: ChartVariantSpec) -> Dict:
    prompt = chart_prompts.build_restyle_prompt(spec.family_hint)
    result = call_images_edit_with_file(client, spec.model, spec.base_image_path, prompt)
    if not hasattr(result, "data") or not result.data or not getattr(result.data[0], "b64_json", None):
        raise RuntimeError("images.edit returned no b64_json payload for variant")

    ts_ms = int(time.time() * 1000)
    safe_family = spec.family_hint.replace(" ", "_").replace("/", "_")
    out_path = spec.out_dir_variants / f"{spec.base_image_path.stem}__restyle__{safe_family}__{ts_ms}.png"
    b64_to_png(result.data[0].b64_json, out_path)

    record = {
        "stage": "variant_restyle",
        "run_id": spec.run_id,
        "timestamp_ms": ts_ms,
        "base_image": str(spec.base_image_path),
        "variant_output": str(out_path),
        "family_hint": spec.family_hint,
        "model": spec.model,
        "prompt": prompt,
        "source_style_image": str(spec.source_style_image),
        "chart_type": spec.chart_type,
        "subject": spec.subject,
    }

    spec.json_dir.mkdir(parents=True, exist_ok=True)
    json_path = spec.json_dir / f"variant__{ts_ms}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def run_charts(args) -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Stage II reads Stage I's per-target JSON logs (stage1_output/json) and writes
    # its content images under <out-subdir>/stage2_output/content.
    in_json_dir = ROOT / args.out_subdir / "stage1_output" / "json"
    out_root = ROOT / args.out_subdir / "stage2_output"
    out_content = out_root / "content"
    json_dir = out_root / "json"
    for p in [out_root, out_content, json_dir]:
        p.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / args.manifest

    primaries: List[Dict] = []
    for jf in sorted(glob(str(in_json_dir / "primary__*.json"))):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                primaries.append(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to load {jf}: {e}", file=sys.stderr)
    if not primaries:
        print(f"[ERROR] No primary__*.json found under {in_json_dir}. Run Stage I first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(primaries)} targets from {in_json_dir}")

    run_id = primaries[0].get("run_id", f"run_{int(time.time())}_{random.randint(1000,9999)}")
    specs: List[ChartVariantSpec] = []
    for rec in primaries:
        specs.extend(make_chart_variant_specs(rec, args.variants_per, args.model, out_content, json_dir))

    variants: List[Dict] = []
    errors: List[Tuple[Path, Exception]] = []
    lock = threading.Lock()
    pbar = tqdm(total=len(specs), desc="Stage II: chart restyles", unit="img") if tqdm else None
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(restyle_chart_variant, client, spec): spec for spec in specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                rec = fut.result()
                with lock:
                    variants.append(rec)
                    if pbar: pbar.update(1)
            except Exception as e:
                with lock:
                    errors.append((spec.base_image_path, e))
                    if pbar: pbar.update(1)
                print(f"[II][ERROR] {spec.base_image_path.name}: {e}", file=sys.stderr)
    if pbar: pbar.close()

    manifest = {
        "run": {
            "run_id": run_id, "timestamp_ms": int(time.time() * 1000),
            "domain": "chart", "stage": "content_generation",
            "output_dir_content": str(out_content.resolve()), "json_dir": str(json_dir.resolve()),
            "model": args.model, "variants_per": args.variants_per, "concurrency": args.concurrency,
            "primary_count": len(primaries), "variant_count": len(variants), "errors_variants": len(errors),
        },
        "primaries": primaries,
        "variants": variants,
        "errors_variants": [{"file": str(p), "error": str(e)} for p, e in errors],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n--- Stage II (charts) Summary ---")
    print(f"Variants: {len(variants)} / {len(specs)}")
    print(f"Manifest written to {manifest_path.resolve()}")


# ================================================================
# FDT DOMAIN
# ================================================================

@dataclass
class FDTVariantSpec:
    base_image_path: Path
    model: str
    family_hint: str
    out_dir_variants: Path
    json_dir: Path
    run_id: str
    source_style_image: Path
    subject: str
    task_type: str
    source_bucket: str


def _fdt_variant_mix(k: int) -> List[str]:
    """
    FDT policy:
    - k >= 2: exactly one minimal family + (k-1) unique families from the full pool.
    - k == 1: one random family from the full pool (no forced minimal or 3D).
    """
    k = max(1, min(k, len(pools.RESTYLE_FAMILIES)))
    if k == 1:
        return [random.choice(pools.RESTYLE_FAMILIES)]
    minimal_choice = random.choice(pools.REQUIRED_MINIMAL_POOL)
    selected = [minimal_choice]
    remaining_needed = k - 1
    remaining_pool = [f for f in pools.RESTYLE_FAMILIES if f != minimal_choice]
    if remaining_needed > len(remaining_pool):
        remaining_needed = len(remaining_pool)
    selected.extend(random.sample(remaining_pool, remaining_needed))
    return selected[:k]


def make_fdt_variant_specs(primary_record: Dict, variants_per: int, model: str,
                           out_dir_variants: Path, json_dir: Path) -> List[FDTVariantSpec]:
    base_image = Path(primary_record["primary_output"])
    k = max(1, min(variants_per, len(pools.RESTYLE_FAMILIES)))
    families = _fdt_variant_mix(k)
    task_type = primary_record.get("task_type", "")
    source_bucket = primary_record.get("source_bucket", "")
    specs: List[FDTVariantSpec] = []
    for fam in families:
        specs.append(
            FDTVariantSpec(
                base_image_path=base_image, model=model, family_hint=fam,
                out_dir_variants=out_dir_variants, json_dir=json_dir,
                run_id=primary_record["run_id"],
                source_style_image=Path(primary_record["style_image"]),
                subject=primary_record.get("subject", ""), task_type=task_type,
                source_bucket=source_bucket,
            )
        )
    return specs


def restyle_fdt_variant(client: OpenAI, spec: FDTVariantSpec) -> Dict:
    prompt = fdt_prompts.build_restyle_prompt(spec.family_hint, spec.task_type)
    result = call_images_edit_with_file(client, spec.model, spec.base_image_path, prompt)
    if not hasattr(result, "data") or not result.data or not getattr(result.data[0], "b64_json", None):
        raise RuntimeError("images.edit returned no b64_json payload for variant")

    ts_ms = int(time.time() * 1000)
    fam_slug = family_to_slug(spec.family_hint)
    base_stem = shorten_stem(spec.base_image_path.stem)
    file_stem = shorten_stem(f"{base_stem}__restyle__{fam_slug}__{ts_ms}", max_len=200)
    out_path = spec.out_dir_variants / f"{file_stem}.png"
    b64_to_png(result.data[0].b64_json, out_path)

    record = {
        "stage": "variant_restyle",
        "run_id": spec.run_id,
        "timestamp_ms": ts_ms,
        "base_image": str(spec.base_image_path.resolve()),
        "variant_output": str(out_path.resolve()),
        "family_hint": spec.family_hint,
        "model": spec.model,
        "prompt": prompt,
        "source_style_image": str(spec.source_style_image.resolve()),
        "subject": spec.subject,
        "task_type": spec.task_type,
        "source_bucket": spec.source_bucket,
    }

    spec.json_dir.mkdir(parents=True, exist_ok=True)
    json_path = spec.json_dir / f"variant__{ts_ms}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def run_fdt(args) -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Read Stage I FDT targets from stage1_output; write content to stage2_output.
    in_base = (ROOT / args.out_subdir / "stage1_output").resolve()
    in_canva_json = in_base / "stylized_canva_images" / "json"
    in_galaxy_json = in_base / "stylized_chart_galaxy_real_images" / "json"

    out_base = (ROOT / args.out_subdir / "stage2_output").resolve()
    out_canva = out_base / "stylized_canva_images"
    out_galaxy = out_base / "stylized_chart_galaxy_real_images"

    def ensure_set_dirs(out_set: Path) -> Tuple[Path, Path]:
        cdir = out_set / "content"
        jdir = out_set / "json"
        for p in [out_set, cdir, jdir]:
            p.mkdir(parents=True, exist_ok=True)
        return cdir.resolve(), jdir.resolve()

    canva_variants_dir, canva_json_dir = ensure_set_dirs(out_canva)
    galaxy_variants_dir, galaxy_json_dir = ensure_set_dirs(out_galaxy)
    manifest_path = (out_base / args.manifest).resolve()

    primaries: List[Dict] = []
    loaded, kept = 0, 0
    for jf in sorted(glob(str(in_canva_json / "primary__*.json")) + glob(str(in_galaxy_json / "primary__*.json"))):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                rec = json.load(f)
                loaded += 1
            if rec.get("task_type") in pools.TASK_TYPES:
                primaries.append(rec)
                kept += 1
        except Exception as e:
            print(f"[WARN] Failed to load {jf}: {e}", file=sys.stderr)
    if kept == 0:
        print("[ERROR] No flowchart/diagram/table primary__*.json found under both json dirs. Run Stage I first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {loaded} primary JSONs, kept {kept} (flowchart/diagram/table).")

    run_id = primaries[0].get("run_id", f"run_mix_{int(time.time())}_{random.randint(1000,9999)}")
    specs: List[FDTVariantSpec] = []
    for rec in primaries:
        bucket = rec.get("source_bucket", "canva")
        if bucket == "galaxy":
            vdir, jdir = galaxy_variants_dir, galaxy_json_dir
        else:
            vdir, jdir = canva_variants_dir, canva_json_dir
        specs.extend(make_fdt_variant_specs(rec, args.variants_per, args.model, vdir, jdir))

    variants: List[Dict] = []
    errors: List[Tuple[Path, Exception]] = []
    lock = threading.Lock()
    pbar = tqdm(total=len(specs), desc="Stage II: FDT restyles", unit="img") if tqdm else None
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(restyle_fdt_variant, client, spec): spec for spec in specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                rec = fut.result()
                with lock:
                    variants.append(rec)
                    if pbar: pbar.update(1)
            except Exception as e:
                with lock:
                    errors.append((spec.base_image_path, e))
                    if pbar: pbar.update(1)
                print(f"[II][ERROR] {spec.base_image_path}: {e}", file=sys.stderr)
    if pbar: pbar.close()

    manifest = {
        "run": {
            "run_id": run_id, "timestamp_ms": int(time.time() * 1000),
            "domain": "fdt", "stage": "content_generation",
            "output_sets": {
                "canva": {"content": str(canva_variants_dir), "json": str(canva_json_dir)},
                "galaxy": {"content": str(galaxy_variants_dir), "json": str(galaxy_json_dir)},
            },
            "manifest_path": str(manifest_path),
            "model": args.model, "variants_per": args.variants_per, "concurrency": args.concurrency,
            "primary_count": len(primaries), "variant_count": len(variants), "errors_variants": len(errors),
        },
        "primaries": primaries,
        "variants": variants,
        "errors_variants": [{"file": str(p), "error": str(e)} for p, e in errors],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n--- Stage II (FDT) Summary ---")
    print(f"Variants: {len(variants)} / {len(specs)}")
    print(f"Manifest path: {manifest_path}")


# ================================================================
# CLI
# ================================================================

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ChartForge Stage II: restyle-based content generation.")
    parser.add_argument("--domain", type=str, required=True, choices=["chart", "fdt"])
    parser.add_argument("--out-subdir", type=str, required=True)
    parser.add_argument("--variants-per", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--model", type=str, default="gpt-image-1")
    parser.add_argument("--manifest", type=str, default="stage2_manifest.json")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    if args.domain == "chart":
        run_charts(args)
    else:
        run_fdt(args)


if __name__ == "__main__":
    main()

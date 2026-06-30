# -*- coding: utf-8 -*-
"""
ChartForge role (reference curation): sample diverse FDT variants as Stage III references.
Sample N diverse F/D/T variants with hard coverage guarantees and single-pass fill.

Goals
-----
1) Coverage-first:
   - Ensure every family is represented at least once.
   - Optionally ensure every (family, task_type) bucket has >= 1 sample if available.

2) Balance & Uniqueness:
   - After coverage, balance across families and rotate task buckets.
   - Enforce uniqueness by primary image (base_image) with a per-primary cap.
   - Never copy the same file twice.

3) Single-pass fill to `--target`:
   - If blocked by per-primary cap, auto-raise the cap up to `--auto-max-per-primary`
     within the same run (no second pass needed).

Inputs
------
Stage-2 manifest (default):
  output/cvpr_chart_restyle/overall_dataset_v1_stage2/manifest_mix_flow_diagram_table_unified.json

Outputs
-------
Destination root (default):
  output/cvpr_chart_restyle/overall_dataset_v1_stage3/reference_images/variants_fdt_10k_as_reference/
  └── selection_summary.json

Usage examples
--------------
# Default paths; target=10k; auto-raise cap up to 3 if needed
python3 sample_stage_3_reference_images_fdt_fill.py

# Explicit paths, force full family+task coverage, strict per-primary=1, but allow auto-raise to 2
python3 sample_stage_3_reference_images_fdt_fill.py \
  --manifest /mnt/.../manifest_mix_flow_diagram_table_unified.json \
  --dest-root /mnt/.../variants_fdt_10k_as_reference \
  --target 10000 \
  --require-all-families \
  --require-all-family-task \
  --max-per-primary 1 \
  --auto-max-per-primary 2
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import hashlib
from collections import defaultdict, Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# -----------------------------
# Canonical vocab
# -----------------------------
SUBJECTS = [
    "Mathematics", "Statistics", "Computer Science", "Physics", "Chemistry",
    "Biology", "Economics", "Business", "Finance", "Marketing",
    "Sociology", "Psychology", "Political Science", "History", "Philosophy",
    "Linguistics", "Education", "Geography", "Environmental Science", "Geology",
    "Engineering", "Public Health", "Medicine", "Law", "Anthropology", "Sports"
]
SUBJECTS_LC = {s.lower() for s in SUBJECTS}

TASK_TYPES = ["flowchart", "diagram", "table"]
TASK_TYPES_LC = {t.lower() for t in TASK_TYPES}

# -----------------------------
# Path remaps if needed
# -----------------------------
PREFIX_REWRITES = []  # add (left, right) pairs if paths moved

def remap_local_path(path_str: str) -> str:
    p = (path_str or "").replace("\\", "/")
    for left, right in PREFIX_REWRITES:
        if p.startswith(left):
            return p.replace(left, right, 1)
    return p

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def shuffle_inplace(x: List[Any]) -> None:
    random.shuffle(x)

class DummyTqdm:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def update(self, n=1): pass

def safe_slug(text: str, max_len: int = 64) -> str:
    if text is None:
        text = ""
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) <= max_len:
        return text
    head = text[: max(16, max_len - 16 - 1)]
    tail = text[-15:]
    return f"{head}_{tail}"

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _normalize_subject_token(token: str) -> Optional[str]:
    if not token: return None
    name = token.replace("_", " ").strip().lower()
    name = _normalize_spaces(name)
    if name in SUBJECTS_LC:
        for s in SUBJECTS:
            if s.lower() == name:
                return s
    return None

def _normalize_task_token(token: str) -> Optional[str]:
    if not token: return None
    name = token.replace("_", " ").strip().lower()
    name = _normalize_spaces(name)
    if name in TASK_TYPES_LC:
        for t in TASK_TYPES:
            if t.lower() == name:
                return t
    return None

def recover_task_subject_from_path(path_like: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Expected stems like:
      ...__A_primary__diagram__Computer_Science__single__1762578619136.png
      ...__A_primary__table__Anthropology__multi__1762578579420.png
    """
    p = (path_like or "")
    stem = Path(p).stem
    parts = stem.split("__A_primary__")
    if len(parts) < 2:
        return (None, None)
    tail = parts[1]
    segs = tail.split("__")
    if len(segs) < 2:
        return (None, None)
    task_token = segs[0]
    subject_token = segs[1]
    task = _normalize_task_token(task_token)
    subj = _normalize_subject_token(subject_token)
    return (task, subj)

# -----------------------------
# Intake and validation
# -----------------------------
def keep_and_fix_fdt_variant(v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Keep if:
      - variant_output and base_image exist on disk
      - task_type & subject exist or recoverable
      - family_hint present
    """
    v_src = remap_local_path(v.get("variant_output", ""))
    v_base = remap_local_path(v.get("base_image", ""))

    if not v_src or not v_base:
        return None
    if not Path(v_src).is_file():
        return None

    family_hint = (v.get("family_hint") or "").strip()
    if not family_hint:
        return None

    task_type = (v.get("task_type") or "").strip()
    subject = (v.get("subject") or "").strip()

    if not task_type or not subject:
        t_from_base, s_from_base = recover_task_subject_from_path(v_base)
        t_from_var, s_from_var = recover_task_subject_from_path(v_src)
        task_type = task_type or t_from_base or t_from_var or ""
        subject   = subject   or s_from_base or s_from_var or ""

    # normalize & validate
    task_norm = _normalize_task_token(task_type.replace(" ", "_")) or _normalize_task_token(task_type) or task_type
    subj_norm = _normalize_subject_token(subject.replace(" ", "_")) or _normalize_subject_token(subject) or subject
    if task_norm.lower() not in TASK_TYPES_LC:
        return None
    if subj_norm.lower() not in SUBJECTS_LC:
        return None

    out = dict(v)
    out["variant_output"] = v_src
    out["base_image"] = v_base
    out["task_type"] = task_norm
    out["subject"] = subj_norm
    out["family_hint"] = family_hint
    return out

# -----------------------------
# Bucketing
# -----------------------------
def build_buckets(variants: List[Dict[str, Any]]):
    fam_task_buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    family_to_task_keys: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for v in variants:
        fam = v["family_hint"]
        tsk = v["task_type"]
        fam_task_buckets[(fam, tsk)].append(v)

    for key in fam_task_buckets:
        shuffle_inplace(fam_task_buckets[key])

    for fam, tsk in fam_task_buckets.keys():
        family_to_task_keys[fam].append((fam, tsk))
    for fam in family_to_task_keys:
        shuffle_inplace(family_to_task_keys[fam])

    families = list(family_to_task_keys.keys())
    shuffle_inplace(families)

    return fam_task_buckets, family_to_task_keys, families

# -----------------------------
# Pick helpers with constraints
# -----------------------------
def can_take(v: Dict[str, Any], used_paths: Set[str], per_primary: Dict[str, int], max_per_primary: int) -> bool:
    src = v["variant_output"]
    if src in used_paths:
        return False
    base = v.get("base_image", "")
    if not base:
        return False
    return per_primary.get(base, 0) < max_per_primary

def take(v: Dict[str, Any], used_paths: Set[str], per_primary: Dict[str, int], selected: List[Dict[str, Any]]):
    used_paths.add(v["variant_output"])
    base = v["base_image"]
    per_primary[base] = per_primary.get(base, 0) + 1
    selected.append(v)

# -----------------------------
# Coverage phases
# -----------------------------
def phase_coverage_families(
    fam_task_buckets, family_to_task_keys, families,
    used_paths, per_primary, selected, max_per_primary
):
    """Ensure at least 1 sample per family (if available)."""
    misses = []
    for fam in families:
        keys = family_to_task_keys.get(fam, [])
        if not keys:
            misses.append(fam)
            continue
        # rotate across that family's task buckets to find a feasible pick
        picked = False
        for _ in range(len(keys)):
            k = keys[0]
            keys = keys[1:] + [k]
            bucket = fam_task_buckets.get(k, [])
            # find first feasible
            while bucket:
                v = bucket.pop()
                if can_take(v, used_paths, per_primary, max_per_primary):
                    take(v, used_paths, per_primary, selected)
                    picked = True
                    break
            if picked:
                break
        if not picked:
            misses.append(fam)
    return misses  # families we couldn't cover (capacity reasons)

def phase_coverage_family_task(
    fam_task_buckets, family_to_task_keys,
    used_paths, per_primary, selected, max_per_primary
):
    """Ensure at least 1 sample per (family, task_type) if available."""
    missed_keys = []
    for key, bucket in fam_task_buckets.items():
        fam, tsk = key
        # try the current bucket without consuming other buckets for this family
        ok = False
        tmp = []
        while bucket:
            v = bucket.pop()
            if can_take(v, used_paths, per_primary, max_per_primary):
                take(v, used_paths, per_primary, selected)
                ok = True
                break
            else:
                tmp.append(v)
        # restore any non-used candidates to preserve future capacity
        while tmp:
            bucket.append(tmp.pop())
        if not ok and bucket:
            missed_keys.append(key)
    return missed_keys

# -----------------------------
# Balanced fill
# -----------------------------
def phase_balanced_fill(
    fam_task_buckets, family_to_task_keys, families, target,
    used_paths, per_primary, selected, max_per_primary
):
    active_fams = [fam for fam in families if family_to_task_keys.get(fam)]
    fam_idx = 0
    fam_task_ptr: Dict[str, int] = {fam: 0 for fam in active_fams}

    with tqdm(total=target-len(selected), desc="Balanced fill", unit="img") if tqdm else DummyTqdm() as pbar:
        while len(selected) < target and active_fams:
            fam = active_fams[fam_idx % len(active_fams)]
            tkeys = family_to_task_keys[fam]
            if not tkeys:
                active_fams.pop(fam_idx % len(active_fams))
                continue
            start_ptr = fam_task_ptr[fam] % len(tkeys)
            local_ptr = start_ptr
            picked = None

            while True:
                key = tkeys[local_ptr]
                bucket = fam_task_buckets.get(key, [])
                tmp = []
                while bucket:
                    v = bucket.pop()
                    if can_take(v, used_paths, per_primary, max_per_primary):
                        picked = v
                        break
                    else:
                        tmp.append(v)
                while tmp:
                    bucket.append(tmp.pop())

                if picked:
                    fam_task_ptr[fam] = (local_ptr + 1) % len(tkeys)
                    break

                local_ptr = (local_ptr + 1) % len(tkeys)
                if local_ptr == start_ptr:
                    # family exhausted
                    active_fams.pop(fam_idx % len(active_fams))
                    picked = None
                    break

            if picked:
                take(picked, used_paths, per_primary, selected)
                pbar.update(1)
            fam_idx += 1

# -----------------------------
# Copy & summary
# -----------------------------
def _build_dst_filename(src_path: Path, family_hint: str, task_type: str, max_filename_len: int = 180) -> str:
    base_stem = src_path.stem
    fam_slug = safe_slug(family_hint, max_len=32)
    t_slug = safe_slug(task_type, max_len=16)

    proposed = f"{base_stem}__fam_{fam_slug}__task_{t_slug}{src_path.suffix}"
    if len(proposed) <= max_filename_len:
        return proposed

    h = hashlib.md5(proposed.encode("utf-8")).hexdigest()[:10]
    short_base = safe_slug(base_stem, max_len=64)
    short_fam = fam_slug[:20]
    short_task = t_slug[:12]
    compressed = f"{short_base}__f_{short_fam}__t_{short_task}__{h}{src_path.suffix}"
    if len(compressed) <= max_filename_len:
        return compressed

    shorter_base = short_base[:40]
    return f"{shorter_base}__{h}{src_path.suffix}"

def infer_source_bucket(path_str: str) -> str:
    p = (path_str or "").replace("\\", "/")
    if "/stylized_canva_images/" in p:
        return "canva"
    if "/stylized_chart_galaxy_real_images/" in p:
        return "galaxy"
    return "unknown"

def copy_selection_and_write_summary(selected: List[Dict[str, Any]], dest_root: Path, summary_name: str = "selection_summary.json") -> None:
    ensure_dir(dest_root)
    family_cnt = Counter()
    task_cnt = Counter()
    family_task_cnt = Counter()
    updated_records: List[Dict[str, Any]] = []

    iterable = tqdm(selected, desc="Copy files", unit="img") if tqdm else selected
    for v in iterable:
        src = remap_local_path(v["variant_output"])
        fam = v["family_hint"]
        tsk = v["task_type"]

        family_cnt[fam] += 1
        task_cnt[tsk] += 1
        family_task_cnt[(fam, tsk)] += 1

        src_path = Path(src)
        dst_name = _build_dst_filename(src_path, fam, tsk, max_filename_len=180)
        dst_path = dest_root / dst_name

        i = 1
        while dst_path.exists():
            name_no_ext = dst_path.stem
            dst_path = dest_root / f"{name_no_ext}__dup{i}{src_path.suffix}"
            i += 1

        shutil.copy2(src_path, dst_path)

        rec = {
            "variant_output": str(src_path.resolve()),
            "base_image": str(Path(v.get("base_image", "")).resolve()),
            "family_hint": fam,
            "task_type": tsk,
            "subject": v.get("subject", ""),
            "dst_path": str(dst_path.resolve()),
            "source_bucket": infer_source_bucket(src),
            "run_id": v.get("run_id", ""),
            "timestamp_ms": v.get("timestamp_ms", ""),
            "model": v.get("model", ""),
        }
        updated_records.append(rec)

    family_task_cnt_str = {f"{k[0]}|||{k[1]}": v for k, v in family_task_cnt.items()}
    summary = {
        "total_selected": len(selected),
        "by_family_hint": dict(family_cnt),
        "by_task_type": dict(task_cnt),
        "by_family_and_task_type": family_task_cnt_str,
        "records": updated_records,
    }

    out_json = dest_root / summary_name
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Saved {len(selected)} images to: {dest_root}")
    print(f"Summary JSON   : {out_json}")
    print(f"Families       : {len(family_cnt)} unique")
    print(f"Task types     : {len(task_cnt)} unique")

# -----------------------------
# Main selection routine
# -----------------------------
def select_with_guarantees(
    variants: List[Dict[str, Any]],
    target: int,
    require_all_families: bool,
    require_all_family_task: bool,
    max_per_primary: int,
    auto_max_per_primary: int,
    seed: int
) -> List[Dict[str, Any]]:

    random.seed(seed)
    fam_task_buckets, family_to_task_keys, families = build_buckets(variants)

    # flatten counts (for quick capacity checks if needed)
    all_buckets_nonempty = any(fam_task_buckets.values())

    used_paths: Set[str] = set()
    per_primary: Dict[str, int] = {}
    selected: List[Dict[str, Any]] = []

    # Adaptive cap loop: try with current cap; if not enough, raise and continue
    cap = max(1, max_per_primary)
    cap_limit = max(cap, auto_max_per_primary)

    while True:
        # Phase 0: if nothing left, break
        if not all_buckets_nonempty:
            break

        # Phase 1: coverage per family
        if require_all_families:
            misses = phase_coverage_families(
                fam_task_buckets, family_to_task_keys, families,
                used_paths, per_primary, selected, cap
            )
            # Misses indicate capacity constraints under current cap; we will rely on later cap-raise
        # Phase 2: coverage per (family, task_type)
        if require_all_family_task:
            _ = phase_coverage_family_task(
                fam_task_buckets, family_to_task_keys,
                used_paths, per_primary, selected, cap
            )

        # Phase 3: balanced fill to target
        if len(selected) < target:
            phase_balanced_fill(
                fam_task_buckets, family_to_task_keys, families, target,
                used_paths, per_primary, selected, cap
            )

        if len(selected) >= target:
            break

        # If we cannot add more with current cap, and cap can be raised, raise and continue.
        # Heuristic: check if ANY bucket still has at least one candidate we rejected ONLY due to cap.
        # For simplicity, we raise cap if there are still items left in buckets and we are short.
        remaining_items = sum(len(lst) for lst in fam_task_buckets.values())
        if remaining_items == 0:
            break

        if cap >= cap_limit:
            # cannot raise further
            break

        cap += 1
        if tqdm:
            print(f"[INFO] Raising per-primary cap to {cap} to reach target {target}...")

        # loop continues with the same working buckets; already-popped items remain popped.
        # That's fine because capacity was the main blocker. If you want to fully re-scan,
        # rebuild buckets from the remaining pool; here we accept a slight under-approximation.

        # To be more aggressive, we can rebuild buckets from the *unselected* pool.
        # Rebuild from scratch using yet-unpicked items:
        remaining_pool = []
        used_set = set(used_paths)  # files we already took
        for v in variants:
            if v["variant_output"] not in used_set:
                remaining_pool.append(v)
        fam_task_buckets, family_to_task_keys, families = build_buckets(remaining_pool)

    return selected[:target]

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=False,
                    default="output/cvpr_chart_restyle/overall_dataset_v1_stage2/manifest_mix_flow_diagram_table_unified.json",
                    help="Unified Stage-2 manifest JSON containing F/D/T variants.")
    ap.add_argument("--dest-root", type=str, required=False,
                    default="output/cvpr_chart_restyle/overall_dataset_v1_stage3/reference_images/variants_fdt_10k_as_reference",
                    help="Destination directory to copy selected images.")
    ap.add_argument("--target", type=int, default=10000, help="Target sample size.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--summary-name", type=str, default="selection_summary.json",
                    help="Summary JSON filename under dest-root.")

    # Coverage & constraints
    ap.add_argument("--require-all-families", action="store_true",
                    help="Guarantee at least one item per family if available.")
    ap.add_argument("--require-all-family-task", action="store_true",
                    help="Guarantee at least one item per (family, task_type) if available.")
    ap.add_argument("--max-per-primary", type=int, default=1,
                    help="Max selected items allowed per base_image.")
    ap.add_argument("--auto-max-per-primary", type=int, default=3,
                    help="Upper bound to auto-raise per-primary cap to fill target in a single run.")

    args = ap.parse_args()
    random.seed(args.seed)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    dest_root = Path(args.dest_root)
    ensure_dir(dest_root)

    print("Loading manifest...")
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept top-level dict with "variants", or a list of variant dicts.
    if isinstance(data, dict) and "variants" in data:
        raw_variants = data.get("variants", [])
    elif isinstance(data, list):
        raw_variants = data
    else:
        raw_variants = []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    raw_variants.extend(v)

    print(f"Variants (raw seen): {len(raw_variants)}")

    variants: List[Dict[str, Any]] = []
    for v in raw_variants:
        fixed = keep_and_fix_fdt_variant(v)
        if fixed is not None:
            variants.append(fixed)

    print(f"Variants (file exists + meta ok): {len(variants)}")
    if len(variants) == 0:
        print("No valid F/D/T variants found (with existing files).", file=sys.stderr)
        sys.exit(1)

    selected = select_with_guarantees(
        variants=variants,
        target=args.target,
        require_all_families=args.require_all_families,
        require_all_family_task=args.require_all_family_task,
        max_per_primary=max(1, args.max_per_primary),
        auto_max_per_primary=max(args.max_per_primary, args.auto_max_per_primary),
        seed=args.seed
    )

    if len(selected) < args.target:
        print(f"WARNING: Copied only {len(selected)} < target {args.target}. "
              f"This indicates global capacity limits even after auto-raising cap.", file=sys.stderr)

    copy_selection_and_write_summary(
        selected=selected,
        dest_root=dest_root,
        summary_name=args.summary_name,
    )

if __name__ == "__main__":
    main()
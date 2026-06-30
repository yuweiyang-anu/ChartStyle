# -*- coding: utf-8 -*-
"""
ChartForge — Stage III: Style-space resampling.

After Stage I/II have produced an initial triplet pool, this stage re-samples
the *style space*: it treats previously generated (and curated) figures as a
unified reference list and re-runs restyling against fresh, deliberately diverse
style families. The goal is to broaden style coverage and rebalance the family
distribution so the final corpus is not dominated by a few popular looks. The
reference list is selected upstream by ``utils/select_stage3_references.py``.

----------------------------------------------------------------------
Unified Stage-3 generator (Charts + Flow/Diagram/Table) using ONE reference list.

- Reference list: summary JSON (records[*].dst_path) and family field records[*].family_hint — DO NOT split; reuse for both tracks.
- Prompts: EXACT originals from your chart pipeline (code I) and F/D/T pipeline (code II). No edits, except removal of any width/height/aspect-related language.
- Variants-per logic:
    * If variants_per == 1: pick one random family from RESTYLE_FAMILIES (avoid reference family if possible).
    * If variants_per >= 2:
        - Avoid the reference family.
        - If the reference family is NOT minimal, include at least one family from REQUIRED_MINIMAL_POOL.
        - If the reference family IS minimal, do NOT force minimal (normal unique random picks).
- This script uses F/D/T images as references; there is no chart-same-type-ratio control.

Outputs (default, all local):
  base_out = output/stage3_output
    - charts/{target,content,json}
    - fdt/{target,content,json}
    - stage3_manifest.json

Example:
  python stage3_style_resampling.py \
    --summary-json output/cvpr_chart_restyle/overall_dataset_v1_stage3/reference_images/variants_fdt_10k_as_reference/selection_summary.json \
    --charts-primary-target 10000 \
    --fdt-primary-target 10000 \
    --variants-per 2 \
    --concurrency 96 \
    --model gpt-image-1
"""

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import threading
import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# --- OpenAI SDK ---
try:
    from openai import OpenAI
except ImportError as e:
    print("Please install the OpenAI SDK: pip install openai", file=sys.stderr)
    raise e


# ---------------------------
# Global constants (EXACT same pools as originals)
# ---------------------------

CHART_TYPES: List[str] = [
    "bar chart", "line chart", "area chart", "scatter plot", "bubble chart",
    "heatmap", "choropleth map", "symbol map", "hexbin plot", "treemap",
    "sunburst chart", "icicle chart", "sankey diagram", "alluvial diagram",
    "chord diagram", "network graph", "tree diagram", "box plot", "violin plot",
    "histogram", "density plot", "ridgeline plot", "dot plot", "pie chart",
    "donut chart", "polar chart", "radar chart", "funnel chart",
    "waterfall chart", "gantt chart", "bullet chart", "parallel coordinates",
    "contour plot", "density map", "timeline chart", "candlestick chart",
]

SUBJECTS: List[str] = [
    "Mathematics", "Statistics", "Computer Science", "Physics", "Chemistry",
    "Biology", "Economics", "Business", "Finance", "Marketing",
    "Sociology", "Psychology", "Political Science", "History", "Philosophy",
    "Linguistics", "Education", "Geography", "Environmental Science", "Geology",
    "Engineering", "Public Health", "Medicine", "Law", "Anthropology", "Sports"
]

INFO_DENSITIES: List[str] = ["standard", "rich", "maximal"]

THREED_FAMILIES: List[str] = [
    "isometric 3D dashboard (soft depth, extruded bars/areas)",
    "3D glassy UI with depth and soft shadows",
    "3D extruded charts with subtle perspective",
    "volumetric lighting and depth layering",
]

REQUIRED_MINIMAL_POOL: List[str] = [
    "python-matplotlib minimal (white background, thin strokes, default system fonts, simple ticks, very light or no grid, no textures)",
    "sketch hand-drawn minimal (extremely simplified, monochrome, rough lines, sparse annotations)"
]
MINIMAL_SET: Set[str] = set(REQUIRED_MINIMAL_POOL)

RESTYLE_FAMILIES: List[str] = [
    "minimalist light theme", "high-contrast dark theme", "pastel soft tints",
    "newspaper greyscale with accent", "data-journalism clean grid",
    "corporate presentation style", "academic journal figure",
    "material design UI", "flat UI dashboard", "blueprint technical",
    "monochrome linework", "hi-vis accessibility palette",
    "retro 1980s infographic", "retro 1990s print", "vintage poster",
    "pop art bold tones", "psychedelic poster", "vaporwave aesthetic",
    "synthwave neon", "Y2K digital gloss", "old cartoon classic", "modern comic halftone",
    "art deco", "art nouveau", "bauhaus swiss mix", "brutalist graphic",
    "neubrutalist UI", "memphis design", "cubism", "surrealism",
    "impressionism", "expressionism", "minimalism", "color field painting",
    "realism", "romanticism", "watercolor illustration", "gouache illustration",
    "acrylic illustration", "pastel illustration", "charcoal and ink", "pen and ink line art",
    "flat vector illustration", "isometric drawing", "pixel art", "risograph print",
    "screen print texture", "woodcut linocut style", "collage mixed media",
    "paper-cut craft", "calligraphic poster", "polaroid instant film",
    "kodak portra 400 look", "ilford hp5 monochrome", "cinematic wide aspect",
    "editorial photography", "analog nostalgia grain", "golden hour photography",
    "blue hour photography", "high-key lighting", "low-key lighting",
    "noir dramatic lighting", "rembrandt lighting", "macro photography",
    "long-exposure light trails", "duotone poster", "gradient mesh poster",
    "holographic foil finish", "newspaper halftone", "kinetic typography",
    "blackletter poster", "slab serif poster", "neo grotesk poster",
    "cartographic terrain tints", "scientific plotting style",
    "magazine cmyk print style", "chalkboard classroom style",
    "glassmorphism gradients", "neumorphism soft UI", "skeuomorphic interface",
    "scanline crt raster", "vhs tape artifact",
    "neon glow poster", "lens flare studio",
    *THREED_FAMILIES,
    *REQUIRED_MINIMAL_POOL,
]

TASK_TYPES = ["flowchart", "diagram", "table"]


# ---------------------------
# Safe filename helpers
# ---------------------------

def safe_slug(text: str, max_len: int = 48) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    norm = unicodedata.normalize("NFKC", text)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", norm).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if len(slug) <= max_len:
        return slug
    h = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
    head = slug[: max(8, max_len - 9)]
    return f"{head}_{h}"

FAMILY_ALIAS = {
    "python-matplotlib minimal (white background, thin strokes, default system fonts, simple ticks, very light or no grid, no textures)": "matplotlib_min",
    "sketch hand-drawn minimal (extremely simplified, monochrome, rough lines, sparse annotations)": "sketch_min",
    "isometric 3D dashboard (soft depth, extruded bars/areas)": "isometric3d_soft",
    "3D glassy UI with depth and soft shadows": "glassy3d_soft",
    "3D extruded charts with subtle perspective": "extruded3d",
    "volumetric lighting and depth layering": "volumetric_depth",
}
def family_to_slug(family: str) -> str:
    return FAMILY_ALIAS.get(family, safe_slug(family, max_len=40))

def shorten_stem(stem: str, max_len: int = 160) -> str:
    if len(stem) <= max_len:
        return stem
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"{stem[: max_len - 9]}_{h}"


# ---------------------------
# I/O helpers
# ---------------------------

def b64_to_png(b64_str: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_str))

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# EXACT original prompts (Charts: code I) — with size/aspect language removed
# ---------------------------

def _complexity_block(enabled: bool) -> str:
    if not enabled:
        return ""
    return """
Complexity guidance from the reference image
-------------------------------------------
• Infer the apparent **visual complexity** of the reference (annotation count, iconography density,
  number of visual layers, micro-grid/texture presence, color stratification, and overall ornamentation).
• Target a **similar complexity** level unless it clearly harms legibility; balance with the chosen info density.
• If single panel: match the reference's density via annotations, subtle callouts, or micro-icons (SVG-like).
• If multi-panel: distribute complexity sensibly across panels; avoid clutter while keeping panels informative.
""".rstrip()

def _title_fidelity_block() -> str:
    return """
Title styling fidelity to the reference
--------------------------------------
• Try to **match the title's typographic style** from the reference: weight, case (ALL CAPS / Title Case / Sentence case),
  tracking/kerning feel, and approximate type family (e.g., geometric sans, humanist, serif-like).
• **Replicate the title's placement** relative to the figure: top-left / centered / top-right / side strip / overlay on a band.
• **Echo decorative treatments** seen in the reference title area:
  – background band or block, underline/hairline rule, corner badge, subtle shadow/glow, gradient strip,
    icon/pictogram near the title, or capsule/pill shape.
• Keep **subtitle styling** aligned with the reference if present (size ratio, weight, spacing, alignment).
• Do not let decorations overpower the data; keep text crisp and readable.
""".rstrip()

def _subject_driven_title_block(enabled: bool, subject: str) -> str:
    if not enabled:
        return ""
    return f"""
Subject-driven title requirements
---------------------------------
• **Do NOT copy** any wording from the reference image's title. Invent a **new** title.
• The title must be **specific and scenario-based** for the subject **{subject}** — not generic.
• You must imagine an appropriate **context** (e.g., a region, a time window, a population group, a metric, an industry),
  and bake that context into the title itself.
• Keep the title concise but informative; optionally include a short subtitle that clarifies scope, units, normalization,
  or methodology — but avoid brand names or trademarks.
• Ensure the title and subtitle remain **consistent** with the synthesized data and the visible panels.
""".rstrip()

def build_chart_primary_prompt(chart_type: str,
                               subject: str,
                               layout_type: str,
                               num_panels: int,
                               info_density: str,
                               panel_chart_types: List[str],
                               complexity_from_reference: bool,
                               subject_driven_titles: bool) -> str:
    layout_desc = (
        f"- Layout: **single panel** (1 chart). Focus on a single coherent visualization."
        if layout_type == "single" else
        f"- Layout: **multi-panel freeform** with **EXACTLY {num_panels}** sub-charts."
    )
    types_desc = ", ".join(panel_chart_types)

    prompt = f"""
You are given a reference style image. Create a **new chart image** that echoes its visual vibe
(typography, palette, stroke weights, background treatment) while **inventing new content**.

Chart parameters
----------------
- Subject: **{subject}**
- Primary chart type: **{chart_type}**
- Info density: **{info_density}**
{layout_desc}

Panel chart types (if multi-panel)
----------------------------------
• Mix across up to 3 types from: {types_desc}

Composition & style
-------------------
• English text only; balanced layout; readable axes, ticks, and legends.
• Encourage subtle SVG-like pictograms or iconography where helpful.
{_title_fidelity_block()}
{_subject_driven_title_block(subject_driven_titles, subject)}
{_complexity_block(complexity_from_reference)}

Output
------
• One flattened raster image; no watermarks.
""".strip()
    return prompt

def build_chart_variant_prompt(family_hint: str) -> str:
    return f"""
You are given a finished chart image. Restyle it to clearly reflect the target visual family: **{family_hint}**.
The goal is to **transform its appearance** within that style family while keeping all information identical.

Keep identical:
---------------
• All data values, scales, chart geometry (bars, lines, areas, axes, legends, annotations).
• All words, labels, and text content (you may move or resize them slightly, but not rewrite or delete).
• Panel count and arrangement.

Style transformation scope:
---------------------------
• Apply the aesthetics and mood of **{family_hint}** — e.g., its color palette, material feel, typography, lighting,
  texture, and visual atmosphere.
• You may reinterpret backgrounds, lighting, color harmony, and figure framing as long as data and wording remain intact.
• Be bold within the boundaries of this style — but do not change content.

Goal:
-----
Produce a chart that visually embodies **{family_hint}**, maintaining every piece of information
while achieving a distinct new visual expression.
Do not crop, distort, or blur data marks.
Output a single clean raster chart; no UI chrome or watermark.
""".strip()


# ---------------------------
# EXACT original prompts (F/D/T: code II) — with size/aspect language removed
# ---------------------------

def _complexity_block_flowchart(level: str, panels: int) -> str:
    if level == "standard":
        nodes = "8–14 nodes total" if panels == 1 else "per panel ~5–9 nodes (total ~10–18)"
        branches = "≥1 decision point with clearly labeled branches"
    elif level == "rich":
        nodes = "14–22 nodes total" if panels == 1 else "per panel ~8–12 nodes (total ~16–28)"
        branches = "≥2 decision points; allow some cross-links and limited loops"
    else:  # maximal
        nodes = "22–36 nodes total" if panels == 1 else "per panel ~10–16 nodes (total ~20–40)"
        branches = "≥3 decision points; allow subflows/loops; ensure merges are clear"
    return f"""
Flowchart complexity
--------------------
• Nodes: {nodes}
• Branching: {branches}
• Use standard shapes: ovals (start/end), rectangles (process), diamonds (decision), parallelograms (I/O).
• Prefer orthogonal connectors; arrowheads visible; avoid overlaps and tangles.
""".rstrip()

def _complexity_block_diagram(level: str, panels: int) -> str:
    if level == "standard":
        comps = "6–10 labeled components" if panels == 1 else "per panel 4–8 components (total ~8–16)"
        edges = "Clear directed edges; minimize crossings"
    elif level == "rich":
        comps = "10–16 labeled components" if panels == 1 else "per panel 6–10 components (total ~12–24)"
        edges = "Allow grouping/swimlanes/region backgrounds; 1–2 cross-group dependencies"
    else:
        comps = "16–26 labeled components" if panels == 1 else "per panel 8–12 components (total ~16–32)"
        edges = "Layered/grouped structure; multiple cross-layer dependencies; legends or small icons permitted"
    return f"""
Diagram complexity
------------------
• Components: {comps}
• Edges & semantics: {edges}
• Use grid alignment and consistent spacing; modular sections may have small titles.
""".rstrip()

def _complexity_block_table(level: str, panels: int) -> str:
    if level == "standard":
        cols = "2–6 columns"
        rows = "2–6 rows per table"
        enrich = "plain header; optional zebra banding; visually suggest sortable columns"
    elif level == "rich":
        cols = "6–10 columns"
        rows = "7–10 rows per table"
        enrich = "grouped headers or filter row; small data-bars/sparklines allowed in 1–2 columns"
    else:
        cols = "10–15 columns"
        rows = "10–25 rows per table"
        enrich = "multi-level headers; row-group footers; inline bars/sparklines/icons; numeric columns right-aligned"
    multi = "" if panels == 1 else "• Multi-panel: each panel is a related table/view; align titles and scale consistently.\n"
    return f"""
Table complexity
----------------
• Columns: {cols}
• Rows: {rows}
• Enrichment: {enrich}
{multi}• Keep columns aligned; consistent units, thousand separators, and numeric precision.
""".rstrip()

def _title_fidelity_block_fdt() -> str:
    return """
Title styling fidelity to the reference
--------------------------------------
• Try to **match the title's typographic style** from the reference: weight, case (ALL CAPS / Title Case / Sentence case),
  tracking/kerning feel, and approximate type family (e.g., geometric sans, humanist, serif-like).
• **Replicate the title's placement** relative to the figure: top-left / centered / top-right / side strip / overlay on a band.
• **Echo decorative treatments** seen in the reference title area:
  – background band or block, underline/hairline rule, corner badge, subtle shadow/glow, gradient strip,
    icon/pictogram near the title, or capsule/pill shape.
• Keep **subtitle styling** aligned with the reference if present (size ratio, weight, spacing, alignment).
• Do not let decorations overpower the data; keep text crisp and readable.
""".rstrip()

def _subject_title_block_fdt(enabled: bool, subject: str) -> str:
    if not enabled:
        return ""
    return f"""
Subject-driven title requirements
---------------------------------
• **Do NOT copy** any wording from the reference image's title. Invent a **new** title.
• The title must be **specific and scenario-based** for the subject **{subject}** — not generic.
• You must imagine an appropriate **context** (e.g., a region, a time window, a population group, a metric, an industry),
  and bake that context into the title itself.
• Keep the title concise but informative; optionally include a short subtitle that clarifies scope, units, normalization,
  or methodology — but avoid brand names or trademarks.
• Ensure the title and subtitle remain **consistent** with the synthesized data and the visible panels.
""".rstrip()

def _diagram_distinction_block() -> str:
    return """
Diagram vs chart distinction (for this task)
--------------------------------------------
• This is a **diagram**, not a numeric chart. Focus on **structure and relationships**.
• Prefer **nodes, groups/swimlanes, arrows, connectors, containment, and hierarchy**.
• **Avoid chart idioms** such as axes, tick marks, bars, lines, areas, pie/donut wedges, scatter points,
  histograms, or any coordinate-based plot unless absolutely essential for clarity.
• Emphasize **logical flow, dependency, or component interaction** rather than quantitative scales.
""".rstrip()

def _panel_block_fdt(layout_type: str, num_panels: int) -> str:
    if layout_type == "single":
        return "- Layout: **single panel** (1 figure)."
    return f"- Layout: **multi-panel freeform** with **EXACTLY {num_panels}** sub-figures; maintain coherent spacing and alignment."

def _generic_style_block_fdt() -> str:
    return """
Composition & global style
--------------------------
• English text only. Balanced layout. Crisp connectors/borders. Legible typography.
• Subtle pictograms/icons allowed if helpful. Avoid clutter and overlapping edges.
• Output must be a flat raster.
""".rstrip()

def build_fdt_primary_prompt(task_type: str,
                             subject: str,
                             layout_type: str,
                             num_panels: int,
                             info_density: str,
                             complexity_from_reference: bool,
                             subject_driven_titles: bool) -> str:
    head = f"""
You are given a reference **style image**. Create a new **{task_type}** that inherits its visual vibe
(typography, palette, stroke weights, background treatment) while inventing new content.
""".strip()

    if task_type == "flowchart":
        cblock = _complexity_block_flowchart(info_density, num_panels)
    elif task_type == "diagram":
        cblock = _complexity_block_diagram(info_density, num_panels)
    else:
        cblock = _complexity_block_table(info_density, num_panels)

    diagram_extra = _diagram_distinction_block() if task_type == "diagram" else ""

    prompt = f"""
{head}

Figure parameters
-----------------
- Subject: **{subject}**
- Info density: **{info_density}**
{_panel_block_fdt(layout_type, num_panels)}

{cblock}
{diagram_extra}

{_generic_style_block_fdt()}
{_title_fidelity_block_fdt()}
{_subject_title_block_fdt(subject_driven_titles, subject)}

Output
------
• One flattened raster image; no watermark.
""".strip()
    if not complexity_from_reference:
        prompt += "\n• Prioritize clarity; do not mimic ornamental complexity from the reference."
    return prompt

def build_fdt_variant_prompt(family_hint: str, task_type: str) -> str:
    noun = {"flowchart": "flowchart", "diagram": "diagram", "table": "table"}.get(task_type, "figure")
    keep_lines = {
        "flowchart": "• Preserve all nodes, labels, connectors, and decision logic.",
        "diagram":   "• Preserve all components, labels, and directed edges/relationships.",
        "table":     "• Preserve all headers, column structure, row ordering, and cell values/labels."
    }.get(task_type, "• Preserve all content elements and structure.")

    return f"""
You are given a finished {noun}. Restyle it to clearly reflect the target visual family: **{family_hint}**.

Keep identical:
---------------
{keep_lines}
• Do not rewrite or delete text content; slight repositioning is allowed for readability.

Style transformation scope:
---------------------------
• Apply **{family_hint}** aesthetics: palette, typography, texture/lighting, framing.
• Backgrounds and atmosphere may change, but information must remain intact.

Output
------
• One clean raster image; no cropping, distortion, or watermark.
""".strip()


# ---------------------------
# Random sampling utils (layout and chart-type selection only)
# ---------------------------

def choose_layout(single_prob: float) -> Tuple[str, int]:
    if random.random() < single_prob:
        return "single", 1
    return "freeform", random.randint(2, 12)

def choose_panel_chart_types(num_panels: int) -> List[str]:
    if num_panels == 1:
        return [random.choice(CHART_TYPES)]
    k = random.choice([1, 2, 3])
    return random.sample(CHART_TYPES, k)


# ---------------------------
# Family canonicalization and exclusion based on JSON field
# ---------------------------

def canonical_family_from_json(name: Optional[str]) -> Optional[str]:
    """
    Map a JSON-provided family name (records[*].family_hint) to one of the canonical RESTYLE_FAMILIES entries.
    Tries exact match, case-insensitive match, then slug/alias match.
    """
    if not name:
        return None
    raw = name.strip()

    # 1) exact match
    for fam in RESTYLE_FAMILIES:
        if fam == raw:
            return fam

    # 2) case-insensitive match
    low = raw.lower()
    for fam in RESTYLE_FAMILIES:
        if fam.lower() == low:
            return fam

    # 3) slug/alias match
    target_slug = family_to_slug(raw)
    for fam in RESTYLE_FAMILIES:
        if family_to_slug(fam) == target_slug:
            return fam

    # fallback: not recognized
    return None


def _choose_variant_families(k: int, exclude_canonical: Set[str]) -> List[str]:
    """
    Choose k families with the following rules:
      - Always avoid families in exclude_canonical.
      - If k == 1: pick one random family from the remaining pool.
      - If k >= 2:
          * If the excluded reference family IS minimal -> do NOT force a minimal family.
          * If the excluded reference family is NOT minimal -> force at least one minimal family.
      - Ensure all chosen families are unique. No 3D requirement.
    """
    k = max(1, min(k, len(RESTYLE_FAMILIES)))
    # Build pool excluding reference family (and any others in exclude set)
    pool = [f for f in RESTYLE_FAMILIES if f not in exclude_canonical]
    if not pool:
        # Fallback: if exclusion wipes everything, revert to full list (still guarantee uniqueness)
        pool = RESTYLE_FAMILIES[:]

    # Determine whether the reference (excluded) family is minimal
    ref_is_minimal = any(f in MINIMAL_SET for f in exclude_canonical)

    # k == 1: simple random from (pool)
    if k == 1:
        return [random.choice(pool)]

    # k >= 2: build a set
    selected: List[str] = []

    if not ref_is_minimal:
        # Force include at least one minimal
        minimal_candidates = [m for m in REQUIRED_MINIMAL_POOL if m not in exclude_canonical]
        if not minimal_candidates:
            minimal_candidates = REQUIRED_MINIMAL_POOL[:]
        minimal_pool = [m for m in minimal_candidates if m in pool] or minimal_candidates
        selected.append(random.choice(minimal_pool))

    # Fill the rest with unique random picks from pool (avoiding duplicates)
    need = k - len(selected)
    available = [f for f in pool if f not in selected]
    if len(available) >= need:
        selected.extend(random.sample(available, need))
    else:
        selected.extend(available)
        need = k - len(selected)
        fallback = [f for f in RESTYLE_FAMILIES if f not in selected]
        if len(fallback) >= need:
            selected.extend(random.sample(fallback, need))
        else:
            remaining = [f for f in RESTYLE_FAMILIES if f not in selected]
            while len(selected) < k and remaining:
                pick = remaining.pop(random.randrange(len(remaining)))
                selected.append(pick)

    # safety: enforce uniqueness and exact length
    uniq = []
    seen = set()
    for f in selected:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
        if len(uniq) == k:
            break
    while len(uniq) < k:
        cand = random.choice(RESTYLE_FAMILIES)
        if cand not in seen:
            uniq.append(cand)
            seen.add(cand)
    return uniq


# ---------------------------
# Dataclasses per track (size/aspect removed)
# ---------------------------

@dataclass
class ChartPrimarySpec:
    style_path: Path
    model: str
    chart_type: str
    subject: str
    layout_type: str
    num_panels: int
    info_density: str
    panel_chart_types: List[str]
    out_dir_primary: Path
    out_dir_variants: Path
    json_dir: Path
    run_id: str
    complexity_from_reference: bool
    subject_driven_titles: bool
    reference_family: Optional[str]  # canonical family from JSON family_hint

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

@dataclass
class FDTPrimarySpec:
    style_path: Path
    model: str
    task_type: str
    subject: str
    layout_type: str
    num_panels: int
    info_density: str
    out_dir_primary: Path
    out_dir_variants: Path
    json_dir: Path
    run_id: str
    complexity_from_reference: bool
    subject_driven_titles: bool
    reference_family: Optional[str]  # canonical family from JSON family_hint

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


# ---------------------------
# Builders
# ---------------------------

def make_chart_primary_specs(style_path: Path, model: str, count: int,
                             out_dir_primary: Path, out_dir_variants: Path, json_dir: Path,
                             run_id: str, single_prob: float,
                             complexity_from_reference: bool, subject_driven_titles: bool,
                             reference_family: Optional[str]) -> List[ChartPrimarySpec]:
    specs: List[ChartPrimarySpec] = []
    for _ in range(count):
        subject = random.choice(SUBJECTS)
        layout_type, num_panels = choose_layout(single_prob)
        info_density = random.choice(INFO_DENSITIES)
        chart_type = random.choice(CHART_TYPES)
        panel_types = choose_panel_chart_types(num_panels)
        specs.append(
            ChartPrimarySpec(
                style_path=style_path,
                model=model,
                chart_type=chart_type,
                subject=subject,
                layout_type=layout_type,
                num_panels=num_panels,
                info_density=info_density,
                panel_chart_types=panel_types,
                out_dir_primary=out_dir_primary,
                out_dir_variants=out_dir_variants,
                json_dir=json_dir,
                run_id=run_id,
                complexity_from_reference=complexity_from_reference,
                subject_driven_titles=subject_driven_titles,
                reference_family=reference_family,
            )
        )
    return specs

def make_fdt_primary_specs(style_path: Path, model: str, count: int,
                           out_dir_primary: Path, out_dir_variants: Path, json_dir: Path,
                           run_id: str, single_prob: float,
                           complexity_from_reference: bool, subject_driven_titles: bool,
                           reference_family: Optional[str]) -> List[FDTPrimarySpec]:
    specs: List[FDTPrimarySpec] = []
    for _ in range(count):
        task = random.choice(TASK_TYPES)
        subject = random.choice(SUBJECTS)
        layout_type, num_panels = choose_layout(single_prob)
        info_density = random.choice(INFO_DENSITIES)
        specs.append(
            FDTPrimarySpec(
                style_path=style_path,
                model=model,
                task_type=task,
                subject=subject,
                layout_type=layout_type,
                num_panels=num_panels,
                info_density=info_density,
                out_dir_primary=out_dir_primary,
                out_dir_variants=out_dir_variants,
                json_dir=json_dir,
                run_id=run_id,
                complexity_from_reference=complexity_from_reference,
                subject_driven_titles=subject_driven_titles,
                reference_family=reference_family,
            )
        )
    return specs


# ---------------------------
# OpenAI images calls
# ---------------------------

def call_images_edit_with_file(client: OpenAI, model: str, image_path: Path, prompt: str):
    try:
        with open(image_path, "rb") as f:
            return client.images.edit(model=model, image=f, prompt=prompt, n=1)
    except AttributeError:
        with open(image_path, "rb") as f:
            return client.images.edits(model=model, image=f, prompt=prompt, n=1)


# ---------------------------
# Execute (Charts)
# ---------------------------

def generate_chart_primary(client: OpenAI, spec: ChartPrimarySpec) -> Dict[str, Any]:
    prompt = build_chart_primary_prompt(
        spec.chart_type,
        spec.subject,
        spec.layout_type,
        spec.num_panels,
        spec.info_density,
        spec.panel_chart_types,
        spec.complexity_from_reference,
        spec.subject_driven_titles
    )
    result = call_images_edit_with_file(client, spec.model, spec.style_path, prompt)
    if not hasattr(result, "data") or not result.data or not getattr(result.data[0], "b64_json", None):
        raise RuntimeError("images.edit returned no b64_json payload for chart primary")

    ts_ms = int(time.time() * 1000)
    style_stem = safe_slug(spec.style_path.stem, max_len=48)
    layout_tag = "single" if spec.layout_type == "single" else "multi"
    safe_chart = safe_slug(spec.chart_type, max_len=24)
    safe_subj = safe_slug(spec.subject, max_len=24)
    file_stem = shorten_stem(f"{style_stem}__A_primary__{safe_chart}__{safe_subj}__{layout_tag}__{ts_ms}", 160)
    out_path = spec.out_dir_primary / f"{file_stem}.png"
    b64_to_png(result.data[0].b64_json, out_path)

    rec: Dict[str, Any] = {
        "stage": "primary",
        "run_id": spec.run_id,
        "timestamp_ms": ts_ms,
        "style_image": str(spec.style_path.resolve()),
        "primary_output": str(out_path.resolve()),
        "chart_type": spec.chart_type,
        "subject": spec.subject,
        "layout": {"layout_type": spec.layout_type, "num_panels": spec.num_panels},
        "panel_chart_types": spec.panel_chart_types,
        "model": spec.model,
        "info_density": spec.info_density,
        "complexity_from_reference": spec.complexity_from_reference,
        "subject_driven_titles": spec.subject_driven_titles,
        "prompt": prompt,
        "track": "charts",
        "reference_family": spec.reference_family,
    }
    jp = spec.json_dir / f"primary__{ts_ms}.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec

def restyle_chart_variant(client: OpenAI, spec: ChartVariantSpec) -> Dict[str, Any]:
    prompt = build_chart_variant_prompt(spec.family_hint)
    result = call_images_edit_with_file(client, spec.model, spec.base_image_path, prompt)
    if not hasattr(result, "data") or not result.data or not getattr(result.data[0], "b64_json", None):
        raise RuntimeError("images.edit returned no b64_json payload for chart variant")
    ts_ms = int(time.time() * 1000)
    fam_slug = family_to_slug(spec.family_hint)
    base_stem = shorten_stem(spec.base_image_path.stem, 160)
    file_stem = shorten_stem(f"{base_stem}__restyle__{fam_slug}__{ts_ms}", 200)
    out_path = spec.out_dir_variants / f"{file_stem}.png"
    b64_to_png(result.data[0].b64_json, out_path)

    rec = {
        "stage": "variant_restyle",
        "run_id": spec.run_id,
        "timestamp_ms": ts_ms,
        "base_image": str(spec.base_image_path.resolve()),
        "variant_output": str(out_path.resolve()),
        "family_hint": spec.family_hint,
        "model": spec.model,
        "prompt": prompt,
        "source_style_image": str(spec.source_style_image.resolve()),
        "chart_type": spec.chart_type,
        "subject": spec.subject,
        "track": "charts"
    }
    jp = spec.json_dir / f"variant__{ts_ms}.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec


# ---------------------------
# Execute (F/D/T)
# ---------------------------

def generate_fdt_primary(client: OpenAI, spec: FDTPrimarySpec) -> Dict[str, Any]:
    prompt = build_fdt_primary_prompt(
        spec.task_type,
        spec.subject,
        spec.layout_type,
        spec.num_panels,
        spec.info_density,
        spec.complexity_from_reference,
        spec.subject_driven_titles
    )
    result = call_images_edit_with_file(client, spec.model, spec.style_path, prompt)
    if not hasattr(result, "data") or not result.data or not getattr(result.data[0], "b64_json", None):
        raise RuntimeError("images.edit returned no b64_json payload for FDT primary")

    ts_ms = int(time.time() * 1000)
    style_stem = safe_slug(spec.style_path.stem, max_len=48)
    layout_tag = "single" if spec.layout_type == "single" else "multi"
    safe_kind = safe_slug(spec.task_type, max_len=16)
    safe_subj = safe_slug(spec.subject, max_len=24)
    file_stem = shorten_stem(f"{style_stem}__A_primary__{safe_kind}__{safe_subj}__{layout_tag}__{ts_ms}", 160)
    out_path = spec.out_dir_primary / f"{file_stem}.png"
    b64_to_png(result.data[0].b64_json, out_path)

    rec: Dict[str, Any] = {
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
        "track": "fdt",
        "reference_family": spec.reference_family,
    }
    jp = spec.json_dir / f"primary__{ts_ms}.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec

def restyle_fdt_variant(client: OpenAI, spec: FDTVariantSpec) -> Dict[str, Any]:
    prompt = build_fdt_variant_prompt(spec.family_hint, spec.task_type)
    result = call_images_edit_with_file(client, spec.model, spec.base_image_path, prompt)
    if not hasattr(result, "data") or not result.data or not getattr(result.data[0], "b64_json", None):
        raise RuntimeError("images.edit returned no b64_json payload for FDT variant")

    ts_ms = int(time.time() * 1000)
    fam_slug = family_to_slug(spec.family_hint)
    base_stem = shorten_stem(spec.base_image_path.stem, 160)
    file_stem = shorten_stem(f"{base_stem}__restyle__{fam_slug}__{ts_ms}", 200)
    out_path = spec.out_dir_variants / f"{file_stem}.png"
    b64_to_png(result.data[0].b64_json, out_path)

    rec = {
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
        "track": "fdt"
    }
    jp = spec.json_dir / f"variant__{ts_ms}.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", type=str, required=True,
                    help="Path to selection_summary.json (records[*].dst_path used as style images; records[*].family_hint used to avoid same family in variants).")
    ap.add_argument("--base-out", type=str, default="output/stage3_output",
                    help="Base output directory (local).")
    ap.add_argument("--charts-primary-target", type=int, default=20000,
                    help="Primaries count for Charts track.")
    ap.add_argument("--fdt-primary-target", type=int, default=20000,
                    help="Primaries count for Flow/Diagram/Table track.")
    ap.add_argument("--variants-per", type=int, default=2,
                    help="Variants per primary. See docstring for minimal-family behavior.")
    ap.add_argument("--single-panel-prob", type=float, default=0.65,
                    help="Probability of single panel layout.")
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--model", type=str, default="gpt-image-1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest-name", type=str, default="stage3_manifest.json")
    # prompt switches (kept for parity with originals; prompts text untouched except size/aspect removal)
    ap.add_argument("--complexity-from-reference", dest="complexity_from_reference",
                    action="store_true", default=True)
    ap.add_argument("--no-complexity-from-reference", dest="complexity_from_reference",
                    action="store_false")
    ap.add_argument("--subject-driven-titles", dest="subject_driven_titles",
                    action="store_true", default=True)
    ap.add_argument("--no-subject-driven-titles", dest="subject_driven_titles",
                    action="store_false")
    args = ap.parse_args()

    random.seed(args.seed)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    # Load reference summary (single pool)
    summary_path = Path(args.summary_json)
    if not summary_path.is_file():
        print(f"ERROR: summary json not found: {summary_path}", file=sys.stderr)
        sys.exit(1)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    records = summary.get("records", [])
    if not records:
        print("ERROR: No records found in summary json.", file=sys.stderr)
        sys.exit(1)

    style_records: List[Dict[str, Any]] = []
    for r in records:
        p = r.get("dst_path", "")
        if p and Path(p).is_file():
            style_records.append(r)
    if not style_records:
        print("ERROR: No existing files in records.dst_path.", file=sys.stderr)
        sys.exit(1)
    print(f"Reference style images available: {len(style_records)}")

    # Prepare outputs
    base_out = Path(args.base_out)
    charts_root = base_out / "charts"
    fdt_root = base_out / "fdt"

    charts_primary_dir = charts_root / "target"
    charts_variants_dir = charts_root / "content"
    charts_json_dir = charts_root / "json"

    fdt_primary_dir = fdt_root / "target"
    fdt_variants_dir = fdt_root / "content"
    fdt_json_dir = fdt_root / "json"

    ensure_dirs(charts_root, charts_primary_dir, charts_variants_dir, charts_json_dir,
                fdt_root, fdt_primary_dir, fdt_variants_dir, fdt_json_dir)

    client = OpenAI(api_key=api_key)
    run_id = f"run_stage3_{int(time.time())}_{random.randint(1000,9999)}"

    # Build Chart primaries
    chart_specs: List[ChartPrimarySpec] = []
    i = 0
    while len(chart_specs) < args.charts_primary_target:
        rec = style_records[i % len(style_records)]
        style_path = Path(rec["dst_path"])
        ref_family_raw = rec.get("family_hint")
        ref_family_canon = canonical_family_from_json(ref_family_raw)
        chart_specs.extend(
            make_chart_primary_specs(
                style_path=style_path,
                model=args.model,
                count=1,
                out_dir_primary=charts_primary_dir,
                out_dir_variants=charts_variants_dir,
                json_dir=charts_json_dir,
                run_id=run_id,
                single_prob=args.single_panel_prob,
                complexity_from_reference=args.complexity_from_reference,
                subject_driven_titles=args.subject_driven_titles,
                reference_family=ref_family_canon
            )
        )
        i += 1

    # Build FDT primaries
    fdt_specs: List[FDTPrimarySpec] = []
    j = 0
    while len(fdt_specs) < args.fdt_primary_target:
        rec = style_records[j % len(style_records)]
        style_path = Path(rec["dst_path"])
        ref_family_raw = rec.get("family_hint")
        ref_family_canon = canonical_family_from_json(ref_family_raw)
        fdt_specs.extend(
            make_fdt_primary_specs(
                style_path=style_path,
                model=args.model,
                count=1,
                out_dir_primary=fdt_primary_dir,
                out_dir_variants=fdt_variants_dir,
                json_dir=fdt_json_dir,
                run_id=run_id,
                single_prob=args.single_panel_prob,
                complexity_from_reference=args.complexity_from_reference,
                subject_driven_titles=args.subject_driven_titles,
                reference_family=ref_family_canon
            )
        )
        j += 1

    # Execute Stage A
    primaries: List[Dict[str, Any]] = []
    errors_primary: List[Tuple[str, Exception]] = []
    lock = threading.Lock()
    total_primary = len(chart_specs) + len(fdt_specs)
    pbar1 = tqdm(total=total_primary, desc="Stage A: primaries", unit="img") if tqdm else None

    def _do_chart_primary(spec: ChartPrimarySpec):
        return generate_chart_primary(client, spec)

    def _do_fdt_primary(spec: FDTPrimarySpec):
        return generate_fdt_primary(client, spec)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = []
        for s in chart_specs: futs.append(pool.submit(_do_chart_primary, s))
        for s in fdt_specs:   futs.append(pool.submit(_do_fdt_primary, s))
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                with lock:
                    primaries.append(rec)
                    if pbar1: pbar1.update(1)
            except Exception as e:
                with lock:
                    errors_primary.append(("primary", e))
                    if pbar1: pbar1.update(1)
                print(f"[A][ERROR] primary: {e}", file=sys.stderr)
    if pbar1: pbar1.close()

    # Build Variant specs (avoid reference/primary family via JSON family_hint; ensure no duplicates)
    variants_specs: List[Any] = []
    for rec in primaries:
        base_image = Path(rec["primary_output"])
        source_style = Path(rec["style_image"])

        exclude_set: Set[str] = set()
        ref_fam = rec.get("reference_family")
        if ref_fam:
            exclude_set.add(ref_fam)

        k = max(1, min(args.variants_per, len(RESTYLE_FAMILIES)))
        families = _choose_variant_families(k, exclude_canonical=exclude_set)

        if rec.get("track") == "charts":
            for fam in families:
                variants_specs.append(
                    ChartVariantSpec(
                        base_image_path=base_image,
                        model=args.model,
                        family_hint=fam,
                        out_dir_variants=charts_variants_dir,
                        json_dir=charts_json_dir,
                        run_id=rec["run_id"],
                        source_style_image=source_style,
                        chart_type=rec["chart_type"],
                        subject=rec["subject"],
                    )
                )
        else:
            ttype = rec.get("task_type", "diagram")
            for fam in families:
                variants_specs.append(
                    FDTVariantSpec(
                        base_image_path=base_image,
                        model=args.model,
                        family_hint=fam,
                        out_dir_variants=fdt_variants_dir,
                        json_dir=fdt_json_dir,
                        run_id=rec["run_id"],
                        source_style_image=source_style,
                        subject=rec["subject"],
                        task_type=ttype
                    )
                )

    # Execute Stage B
    variants: List[Dict[str, Any]] = []
    errors_variants: List[Tuple[str, Exception]] = []
    total_variants = len(variants_specs)
    pbar2 = tqdm(total=total_variants, desc="Stage B: variants", unit="img") if tqdm else None

    def _do_chart_variant(spec: ChartVariantSpec):
        return restyle_chart_variant(client, spec)

    def _do_fdt_variant(spec: FDTVariantSpec):
        return restyle_fdt_variant(client, spec)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs2 = []
        for s in variants_specs:
            if isinstance(s, ChartVariantSpec):
                futs2.append(pool.submit(_do_chart_variant, s))
            else:
                futs2.append(pool.submit(_do_fdt_variant, s))
        for fut in as_completed(futs2):
            try:
                rec = fut.result()
                with lock:
                    variants.append(rec)
                    if pbar2: pbar2.update(1)
            except Exception as e:
                with lock:
                    errors_variants.append(("variant", e))
                    if pbar2: pbar2.update(1)
                print(f"[B][ERROR] variant: {e}", file=sys.stderr)
    if pbar2: pbar2.close()

    # Manifest
    manifest = {
        "run": {
            "run_id": run_id,
            "timestamp_ms": int(time.time() * 1000),
            "reference_summary_json": str(summary_path.resolve()),
            "reference_count_existing": len(style_records),
            "charts": {
                "primary_dir": str(charts_primary_dir.resolve()),
                "variants_dir": str(charts_variants_dir.resolve()),
                "json_dir": str(charts_json_dir.resolve()),
                "primary_target": args.charts_primary_target,
            },
            "fdt": {
                "primary_dir": str(fdt_primary_dir.resolve()),
                "variants_dir": str(fdt_variants_dir.resolve()),
                "json_dir": str(fdt_json_dir.resolve()),
                "primary_target": args.fdt_primary_target,
            },
            "model": args.model,
            "variants_per": args.variants_per,
            "single_panel_prob": args.single_panel_prob,
            "complexity_from_reference": args.complexity_from_reference,
            "subject_driven_titles": args.subject_driven_titles,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "errors_primary": len(errors_primary),
            "errors_variants": len(errors_variants),
            "counts": {
                "primary_total": len(primaries),
                "variants_total": len(variants),
            }
        },
        "primaries": primaries,
        "variants": variants,
        "errors_primary": [{"stage": s, "error": str(e)} for s, e in errors_primary],
        "errors_variants": [{"stage": s, "error": str(e)} for s, e in errors_variants],
    }

    manifest_path = Path(args.base_out) / args.manifest_name
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n--- Summary ---")
    print(f"Chart primaries: {sum(1 for p in primaries if p.get('track')=='charts')} / target {args.charts_primary_target}")
    print(f"FDT primaries  : {sum(1 for p in primaries if p.get('track')=='fdt')} / target {args.fdt_primary_target}")
    print(f"Variants total : {len(variants)} (expected { (args.charts_primary_target + args.fdt_primary_target) * args.variants_per })")
    print(f"Manifest       : {manifest_path.resolve()}")

if __name__ == "__main__":
    main()
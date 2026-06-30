# -*- coding: utf-8 -*-
"""
Stage I / Stage II prompt builders for the **chart** domain.

These functions assemble the exact prompts sent to the image generator
(GPT-Image via the Images API) when synthesizing chart targets (Stage I) and
restyling them into new style families (Stage II). The prompt text is preserved
verbatim from the original ChartForge pipeline.

- ``build_primary_prompt``  : Stage I — reference-driven chart target generation.
- ``build_restyle_prompt``  : Stage II — restyle a finished chart into a family.

The leading ``_*_block`` helpers are reusable prompt fragments that are spliced
into the primary prompt.
"""

from typing import List


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
    """Strong guidance to match the reference title's styling and position."""
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
• Keep text crisp and readable.
""".rstrip()

def _subject_driven_title_block(enabled: bool, subject: str) -> str:
    """
    Require the model to invent a subject-grounded, concrete scenario title itself
    (NO copying user/reference title; NO hardcoded words from us).
    """
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

def build_primary_prompt(chart_type: str,
                         subject: str,
                         layout_type: str,
                         num_panels: int,
                         style_filename: str,
                         info_density: str,
                         panel_chart_types: List[str],
                         complexity_from_reference: bool,
                         subject_driven_titles: bool) -> str:
    """Construct the prompt for Stage A."""
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


def build_restyle_prompt(family_hint: str):
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
Do not blur data marks.
Output a single clean raster chart; no UI chrome or watermark.
""".strip()

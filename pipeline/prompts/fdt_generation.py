# -*- coding: utf-8 -*-
"""
Stage I / Stage II prompt builders for the **flowchart / diagram / table (FDT)** domain.

These functions assemble the exact prompts sent to the image generator when
synthesizing structural-visualization targets (Stage I) and restyling them into
new style families (Stage II). The prompt text is preserved verbatim from the
original ChartForge pipeline.

- ``build_primary_prompt``  : Stage I — reference-driven FDT target generation.
- ``build_restyle_prompt``  : Stage II — restyle a finished FDT into a family.

The complexity helpers are type-specific (flowchart / diagram / table) and adapt
to the requested information density and panel count.
"""


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
• Do not let decorations overpower the data; keep text crisp and readable.
""".rstrip()

def _subject_title_block(enabled: bool, subject: str) -> str:
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

# >>>>>>>>>>>>>>>> Explicit distinction for diagrams vs charts <<<<<<<<<<<<<<<<
def _diagram_distinction_block() -> str:
    """
    Add explicit constraints so 'diagram' outputs are structurally/relationally focused,
    not numeric chart renders.
    """
    return """
Diagram vs chart distinction (for this task)
--------------------------------------------
• This is a **diagram**, not a numeric chart. Focus on **structure and relationships**.
• Prefer **nodes, groups/swimlanes, arrows, connectors, containment, and hierarchy**.
• **Avoid chart idioms** such as axes, tick marks, bars, lines, areas, pie/donut wedges, scatter points,
  histograms, or any coordinate-based plot unless absolutely essential for clarity.
• Emphasize **logical flow, dependency, or component interaction** rather than quantitative scales.
""".rstrip()
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def _panel_block(layout_type: str, num_panels: int) -> str:
    if layout_type == "single":
        return "- Layout: **single panel** (1 figure)."
    return f"- Layout: **multi-panel freeform** with **EXACTLY {num_panels}** sub-figures; maintain coherent spacing and alignment."

def _generic_style_block() -> str:
    return """
Composition & global style
--------------------------
• English text only. Balanced layout. Crisp connectors/borders. Legible typography.
• Subtle pictograms/icons allowed if helpful. Avoid clutter and overlapping edges.
• Output must be a flat raster; do not crop.
""".rstrip()

def build_primary_prompt(task_type: str,
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

    # Explicit diagram-vs-chart guidance only for diagrams
    diagram_extra = _diagram_distinction_block() if task_type == "diagram" else ""

    prompt = f"""
{head}

Figure parameters
-----------------
- Subject: **{subject}**
- Info density: **{info_density}**
{_panel_block(layout_type, num_panels)}

{cblock}
{diagram_extra}

{_generic_style_block()}
{_title_fidelity_block()}
{_subject_title_block(subject_driven_titles, subject)}

Output
------
• One flattened raster image; no watermark.
""".strip()
    if not complexity_from_reference:
        prompt += "\n• Prioritize clarity; do not mimic ornamental complexity from the reference."
    return prompt


def build_restyle_prompt(family_hint: str, task_type: str) -> str:
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
• One clean raster image; no cropping or watermark.
""".strip()

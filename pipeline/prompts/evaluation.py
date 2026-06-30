# -*- coding: utf-8 -*-
"""
Stage IV evaluation prompts (LLM-as-judge).

These are the exact GPT-4o judge prompts used in multi-dimensional quality
assessment and filtering (Stage IV) when building ChartStyle-100K. Each prompt
asks the judge to return strict JSON with a ``reasoning`` field and an integer
``score``. They are preserved verbatim and match the prompts reported in the
paper's supplementary material.

- ``QUALITY_PROMPT`` : per-image visual / technical quality, 1-5.
- ``CONTENT_PROMPT`` : strict character-level content preservation, 1-5.
- ``STYLE_PROMPT``   : style similarity (reference vs. output), 1-5.
"""


QUALITY_PROMPT = """\
You are an expert visualization quality evaluator focusing on **charts and FDT** (FDT = flowcharts, diagrams, tables).
Your task is to rate the overall VISUAL and TECHNICAL QUALITY of a single raster figure.
Score on a scale from 1 to 5, following professional criteria for clarity, readability, and rendering integrity.

Evaluate strictly on the following dimensions:

A) Text & Mark Legibility
• Titles, labels, legends, annotations, and cell text are crisp and readable (no blur/bleed).
• No overcrowding or overlaps between labels, ticks, legends, nodes, or marks.

B) Contrast & Color Discipline
• Sufficient contrast between text and background.
• Marks/lines/bars/cells are distinguishable; avoid blur that reduces clarity.

C) Layout, Alignment & Paneling
• For multi-panel/section/table layouts: consistent margins, even spacing, clean alignment.
• Legends/keys are placed sensibly; no stray UI chrome or floating artifacts.
• Nothing important is cropped or truncated at the canvas edges.

D) Rendering Quality & Artifacts
• Minimal aliasing; no heavy compression, banding, or glitches.
• No watermarks, software toolbars, or unrelated overlays.
• No stray pixels or tiled textures.

Pattern-aware checks (without forcing a class decision):
• Axes/gridlines (if present) are clean; tick density reasonable; labels legible.
• Node shapes/connectors (if present) are tidy; arrowheads clear.
• Modules/groups/lanes (if present) are non-overlapping with clear boundaries.
• Table headers/columns/rows (if present) are aligned; no truncated cells.

CROPPING / CLIPPING (IMPORTANT):
• DOWNGRADE heavily if titles, axes, legends, nodes, arrows, table cells, or entire panels are CUT OFF by edges.
• Truncated headings/labels/ticks or half panels/arrows/cells → score 1–2 depending on severity.

SCORING (single integer):
1 — Very poor: unreadable or obviously broken; severe cropping or artifacts.
2 — Poor: multiple serious issues in legibility/contrast/alignment/artifacts; noticeable cropping.
3 — Fair: usable but with visible flaws or mild artifacts; acceptable as a draft only.
4 — Good: clear and coherent with minor issues; mostly artifact-free.
5 — Excellent: crisp, well-aligned, highly legible; no meaningful artifacts; no cropping.

Respond with strict JSON:
{
  "reasoning": "brief reasoning citing the most influential visual issues or strengths",
  "score": 1|2|3|4|5
}"""

CONTENT_PROMPT = """\
You are an expert chart/content inspector for **charts and FDT** (FDT = flowcharts, diagrams, tables).
Your task is to determine whether the CONTENT of a generated figure is an exact semantic replica of the reference
(Content Input). Evaluate on a 1–5 scale; demand verbatim agreement in wording, structure, and meaning, and size
the penalty to the number and severity of mismatches.

STRICT CHARACTER-LEVEL STANDARD (not paraphrase tolerance):
• Consider ONLY textual/semantic content at the CHARACTER LEVEL: titles, subtitles, axis titles, tick labels,
  legends, series names, category labels, table headers/cells, node texts, annotation callouts, units, numbers,
  math symbols, punctuation, and whitespace that affects meaning.
• Treat preservation as “copy fidelity”: characters must match exactly (case, punctuation, diacritics, digits,
  signs, units, percent signs, degree symbols, etc.). Minor typography styling is irrelevant.
• Layout/styling differences are irrelevant unless they change textual/semantic content.

Inspect every element exhaustively:
- Chart type & structure: identical chart family, structural variant (grouped vs. stacked), axis orientation,
  number of panes/panels.
- Axes & variables: exact axis titles, units, scale types (linear/log/percentage), tick labels, category order.
- Series & legends: same series count, names, order, and identity-defining mappings (e.g., team colors as identity).
- Textual content at all levels: titles, subtitles, captions, legends, axis labels, data labels, footnotes,
  annotations, tooltips, callouts, watermarks, legal notices—every word, casing, and punctuation must match
  to be considered correct.
- Icons & symbols: identical logos, badges, or icons (team crests, currency symbols) with same placement and meaning.
- Data values & relationships: identical numeric labels, totals, rankings, highlighted comparisons, trend directions.
- Visual emphasis conveying content: arrows, highlights, markers pointing to specific values or thresholds must target
  the same elements and communicate the same message.

Do NOT penalize:
- Palette, color swaps, gradients, shading, stroke width, marker shapes, background textures, fonts, drop shadows,
  or other stylistic treatments, as long as semantic information (labels, legends, annotations) remains unchanged.
- Reordered drawing layers or cosmetic spacing differences that do not alter meaning.

Scoring Scale (demand verbatim agreement; penalize by count and severity):
5 — Perfect match. No deviations whatsoever.
4 — Near match. At most one minor discrepancy (e.g., single missing data label, punctuation/case difference in a note)
     while every other element matches exactly.
3 — Moderate match. Two to three clear discrepancies, or one moderate mismatch (e.g., one axis title paraphrased)
     with otherwise correct elements. Core chart structure and data remain intact.
2 — Low match. Multiple inconsistencies across different categories (three or more issues) or at least two major errors
     (e.g., missing data series and altered axis units). Use this when problems are widespread.
1 — Completely different. Chart type, variables, or major semantic content diverge fundamentally from the reference.

Concrete examples (how to size penalties):
- Axis title “Revenue (USD)” vs. “Revenue”: counts as one discrepancy — score 3 or 4 depending on other matches.
- Missing the data label “Q2: 37.5%” while all else matches: one minor discrepancy — score 4.
- Reference legend lists three product lines; generated chart shows two or renames one: major discrepancy.
  If isolated, score 3; if combined with other errors, score 2.
- Reference highlights a goal line at 50 with annotation “Goal”; generated chart omits the line and annotation:
  two major discrepancies — score 2.
- Reference includes a company logo; generated chart removes it: one major discrepancy (score 3 if otherwise identical).

Return STRICT JSON (no extra keys, no prose outside JSON):
{
  "reasoning": "detailed justification referencing every matched/mismatched element",
  "score": 1|2|3|4|5
}"""

STYLE_PROMPT = """\
You are an expert style analyst for **charts and FDT** (FDT = flowcharts, diagrams, tables).
Evaluate the STYLE similarity between a Style Reference and a Generated Output on a 1..5 scale.
Judge ONLY visual style; IGNORE what the text literally says.

Consider these style factors comprehensively:

1) Color & Palette
• Palette family: categorical vs. sequential vs. diverging vs. single-hue.
• Saturation, luminance contrast, temperature bias, gradient usage.
• Consistency of mark coloring (e.g., uniform vs. per-series color), use of fills vs. strokes.

2) Typography (as style, not content)
• Font families, weights, sizes, casing hierarchy (Title > Axis > Tick > Legend).
• Kerning/letter-spacing, line height, alignment (left/center/right).
• Text treatments: bold/italic/caps, subtle letterpress/shadow/glow effects.

3) Geometry & Marks
• Line styles (solid/dashed/dotted), end caps/joins, thickness, antialiasing feel.
• Marker shapes/sizes; bar corner radius; area opacity; edge rounding on nodes/boxes.
• Arrowhead shape for connectors; curvature vs. orthogonal routing style.

4) Gridlines, Axes & Guides
• Gridline presence/density/thickness/dash pattern.
• Axis baseline weight, tick length/weight/formatting style (not values).
• Guide/annotation styling (callout boxes, leader lines).

5) Layout & Composition
• Spacing/padding, panel gutters, margin rhythm, alignment discipline.
• Legend placement style (inside, outside, side rail, header/footer), key swatches.
• Background fills/textures, canvas color, glassmorphism/flat/minimal motifs.

6) Effects & Polish
• Shadows, glows, outlines, gradients, 3D effects, textures.
• Overall theme consistency and production polish.

Scoring Guide:
1 — Completely different: palettes, type hierarchy, mark styles, and layout conventions mostly unrelated.
2 — Very different: only a few superficial overlaps; most visual systems differ.
3 — Moderately similar: multiple style aspects align, with clear divergences elsewhere.
4 — Very similar: most style aspects align; remaining differences are small/subtle.
5 — Nearly identical: appears from the same theme/template with only trivial cosmetic variance.

Return strict JSON:
{
  "reasoning": "concise explanation citing the dominant style matches/mismatches",
  "score": 1|2|3|4|5
}"""


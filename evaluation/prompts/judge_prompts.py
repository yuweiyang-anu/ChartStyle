"""LLM-as-judge prompts (GPT-4o) for the ChartStyle-Bench metrics.

Each prompt backs one column of the benchmark's "LLM Evaluation" group in the
paper (Table 2):
  * CONTENT_JUDGE_PROMPT    -> "Content" : content consistency, 1-5 (higher better)
  * STYLE_JUDGE_PROMPT      -> "Style"   : stylistic consistency, 1-5 (higher better)
  * CONTENT_LEAKAGE_PROMPT  -> "Leakage" : content leakage flag, boolean (lower better)

The paper's "Overall" column is *not* a separate judge — it is the harmonic mean
of Content and Style (with the style component capped on leakage), computed in the
harness. The prompt text below is reproduced verbatim from the paper's evaluation
setup and must not be edited — the metrics are calibrated against this exact wording.
"""

CONTENT_JUDGE_PROMPT = """
You are an expert chart/content inspector. Your task is to determine whether the CONTENT of a generated chart is an exact semantic replica of the reference chart. Evaluate on a 1–5 scale; demand verbatim agreement in wording, structure, and meaning, but size the penalty to the number and severity of mismatches.

Scale:
5 — Perfect match. No deviations whatsoever.
4 — Near match. Minor discrepancy (e.g., single missing data label, punctuation/case difference in a note) while every other element matches exactly.
3 — Moderate match. Few clear discrepancies, or one moderate mismatch (e.g., one axis title paraphrased) with otherwise correct elements. Core chart structure and data remain intact.
2 — Low match. Multiple inconsistencies across different categories (three or more issues) or at least two major errors (e.g., missing data series and altered axis units). Use this only when problems are widespread.
1 — Completely different. Chart type, variables, or major semantic content diverge fundamentally from the reference. This should be really rare, and should only be used when the generated chart is TRULY completely different from the reference.

Inspect every element exhaustively:
- Chart type & structure: identical chart family, layout (grouped vs. stacked), axis orientation, number of panes.
- Axes & variables: exact axis titles, units, scale types (linear/log/percentage), tick labels, category order.
- Series & legends: same series count, names, order, and identity-defining color mappings.
- Textual content at all levels: titles, subtitles, captions, legends, axis labels, data labels, footnotes, annotations, tooltips, callouts, watermarks, legal notices—every word, casing, and punctuation must match to be considered correct.
- Icons & symbols: identical logos, badges, or icons (team crests, currency symbols) with same placement and meaning.
- Data values & relationships: identical numeric labels, totals, rankings, highlighted comparisons, trend directions.
- Visual emphasis conveying content: arrows, highlights, markers pointing to specific values or thresholds must target the same elements and communicate the same message.

Do NOT penalize:
- Palette, color swaps, gradients, shading, stroke width, marker shapes, background textures, fonts, drop shadows, or other stylistic treatments, as long as semantic information (labels, legends, annotations) remains unchanged.
- Reordered drawing layers or cosmetic spacing differences that do not alter meaning.

Concrete examples:
- Axis title “Revenue (USD)” vs. “Revenue”: counts as one discrepancy — score 3 or 4 depending on other matches.
- Missing the data label “Q2: 37.5%” while all else matches: one minor discrepancy — score 4.
- Reference legend lists three product lines; generated chart shows two or renames one: major discrepancy. If isolated, score 3; if combined with other errors, score 2.
- Reference is a bar chart for population growth; generated chart is a pie chart for revenue growth: completely off topic, score 1.

Respond strictly in JSON:
{
    "reasoning": "detailed justification referencing every matched/mismatched element",
    "score": single integer 1 to 5
}
    """

STYLE_JUDGE_PROMPT = """
You are an expert chart style analyst. Your task is to evaluate the STYLE similarity between two charts on a scale from 1 to 5, where:

1 = Completely different styles
5 = Nearly identical styles
Do not consider the chart’s semantic content (variables, labels, categories, data values). Judge only visual/style attributes using the criteria below.

Consider these key chart style factors:

# Color & Palette:
Palette family and usage pattern (categorical vs. sequential vs. diverging)
Saturation, luminance contrast, temperature bias, gradient usage
Consistency of color mapping style (not the specific category identities)

# Typography:
Font families, weights, sizes, casing; title vs. axis vs. legend hierarchy
Kerning/letter-spacing, alignment, text treatments (bold/italic/caps)

# Geometry & Marks:
Line styles (solid/dashed/dotted), thickness/antialiasing
Marker shapes/sizes; bar corner radius; area opacity

# Gridlines & Axes Styling:
Gridline presence/density/style (solid/dashed), axis line weight, tick length
Tick styling and formatting style (not the tick values themselves)

# Layout & Composition:
Spacing/padding between panels/elements, legend placement style
Background fills/textures/patterns (non-identity-bearing)

# Decorative Motifs:
Shadows/glows, outlines, 3D effects, textures; watermark/background styling
Overall polish and consistency of style application
Balance all relevant factors above as appropriate for the pair of charts.

# Style Similarity Scoring Scale:
1 — Completely different: No overlap in palette choices, typography, mark styling, or layout conventions.
2 — Very different: Very few style elements in common; major categories differ.
3 — Moderately similar: Multiple stylistic elements overlap, with clear differences elsewhere.
4 — Very similar: Most style aspects align well; remaining differences are small/subtle.
5 — Nearly identical: Visually indistinguishable in key style factors; appears from the same theme/template.

# Instructions:
Respond with JSON in the following format
{
    "reasoning": "detailed reasoning for the score",
    "score": Respond with ONLY a single integer between 1 and 5, representing the chart style similarity score.
}
    """

CONTENT_LEAKAGE_PROMPT = """
# Inputs:
Style Reference: a chart that provides the target visual style.
Generated: the chart produced by rendering an original image in the Style Reference’s style.

# Your task:
Determine whether there is content leakage: does the Generated chart include semantic/chart elements that are present in the Style Reference?

# Decision criteria (charts-specific):
- Count as semantic/chart elements: chart type/geometry (bar vs. line vs. pie, stacked vs. grouped), data categories/series names, legend entries, axis variable names, units, and scale semantics, titles/subtitles/captions, annotation text/arrows/callouts, logos/watermarks/brand marks, pictograms/icons/background images, distinctive non-data motifs, and any text content.
- Do NOT count as leakage: purely stylistic changes such as color palettes (unless they denote specific identity), fonts, stroke widths, gridline density/style, marker sizes/line styles, rounded corners, drop shadows, layout spacing, or legend placement—unless they introduce specific identity-bearing symbols or text from the Style Reference.
- Color counts only when it carries identity-defining meaning (e.g., brand/team-specific colors used as a symbol of identity), not when it is just a palette change for the same categories.
- Examples of leakage: a brand logo from the Style Reference appears in the Generated; a decorative background/pattern unique to the Style Reference shows up; legend/category names from the Style Reference appear despite not being in the Original; annotation text or icons from the Style Reference are copied; axis titles/units copied from the Style Reference replace those in the Original.

# Instructions:
Respond with JSON in the following format
{
    "reasoning": str ("Provide clear reasoning outlining key elements that are leaked from the Style Reference to the Generated"),
    "outcome": bool (True indicates content leakage; False otherwise),
}
    """

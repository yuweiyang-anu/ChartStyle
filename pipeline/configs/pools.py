# -*- coding: utf-8 -*-
"""
Sampling pools for the ChartForge data-generation pipeline.

These pools define the controlled randomization space used in Stage I
(reference-driven target generation) and Stage II (restyle-based content
generation). They are shared verbatim across the chart and flowchart/diagram/
table (FDT) generators so that both domains draw from an identical style space.

Pools
-----
- ``CHART_TYPES``           : 36 fine-grained chart types (chart domain only).
- ``TASK_TYPES``            : coarse FDT families (flowchart / diagram / table).
- ``SUBJECTS``              : 26 academic / professional subject domains.
- ``INFO_DENSITIES``        : information-density levels per figure.
- ``THREED_FAMILIES``       : 3D-leaning restyle families.
- ``REQUIRED_MINIMAL_POOL`` : minimal baseline styles (matplotlib / hand-drawn).
- ``RESTYLE_FAMILIES``      : full restyle style-family pool (Stage II).
- ``FAMILY_ALIAS``          : short filesystem-safe aliases for long family names.
"""

from typing import Dict, List


# Fine-grained chart types (chart domain only).
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

# Coarse structural families (FDT domain only).
TASK_TYPES: List[str] = ["flowchart", "diagram", "table"]

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

# Short, filesystem-safe aliases for long family names (used by the FDT generator
# to keep output filenames within OS length limits).
FAMILY_ALIAS: Dict[str, str] = {
    "python-matplotlib minimal (white background, thin strokes, default system fonts, simple ticks, very light or no grid, no textures)": "matplotlib_min",
    "sketch hand-drawn minimal (extremely simplified, monochrome, rough lines, sparse annotations)": "sketch_min",
    "isometric 3D dashboard (soft depth, extruded bars/areas)": "isometric3d_soft",
    "3D glassy UI with depth and soft shadows": "glassy3d_soft",
    "3D extruded charts with subtle perspective": "extruded3d",
    "volumetric lighting and depth layering": "volumetric_depth",
}

"""Load the ChartStyle-Bench benchmark directly from the Hugging Face Hub.

The benchmark is published as a Parquet dataset with embedded images at
``ChartFoundation/ChartStyleBench`` with the schema:

    id              str    sample identifier ("001"-"300")
    style_reference image  style reference visualization
    content_image   image  content visualization to be restyled
    content_type    str    chart / flowchart / diagram / table

This module wraps ``datasets.load_dataset`` and yields lightweight sample dicts
with decoded PIL images, so the harness never touches the Hub schema directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PIL import Image


@dataclass
class BenchSample:
    sample_id: str
    style_reference: Image.Image
    content_image: Image.Image
    content_type: Optional[str] = None


def load_bench(
    repo_id: str = "ChartFoundation/ChartStyleBench",
    split: str = "test",
    limit: Optional[int] = None,
    token: Optional[str] = None,
) -> List[BenchSample]:
    """Return ChartStyle-Bench samples with decoded PIL images.

    Args:
        repo_id: HF dataset repo (default: the public ChartStyle-Bench).
        split:   dataset split to load.
        limit:   if set, only the first N samples are returned (useful for smoke tests).
        token:   HF access token (only needed for private/gated repos).
    """
    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split, token=token)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    samples: List[BenchSample] = []
    for i, row in enumerate(ds):
        sample_id = str(row.get("id", row.get("sample_id", f"{i:03d}")))
        style = row["style_reference"]
        content = row["content_image"]
        samples.append(
            BenchSample(
                sample_id=sample_id,
                style_reference=style.convert("RGB") if isinstance(style, Image.Image) else style,
                content_image=content.convert("RGB") if isinstance(content, Image.Image) else content,
                content_type=row.get("content_type"),
            )
        )
    return samples

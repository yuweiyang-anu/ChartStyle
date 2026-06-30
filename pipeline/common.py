# -*- coding: utf-8 -*-
"""
Shared helpers for the ChartForge generation stages (Stage I & Stage II).

This module collects the small, prompt-agnostic utilities that the chart and
FDT generators both rely on: image discovery, base64→PNG writing, OpenAI Images
API invocation, and filesystem-safe filename slugging. The logic is preserved
verbatim from the original pipeline scripts.
"""

import base64
import hashlib
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    print("Please install the OpenAI SDK: pip install openai", file=sys.stderr)
    raise e


# --------------------------------
# Image discovery
# --------------------------------

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def discover_style_images(images_dir: Path, image_start: int = 0,
                          image_end: Optional[int] = None) -> List[Path]:
    """Flat (non-recursive) reference discovery used by the chart generator."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: _natural_key(p.name))
    if image_end is None:
        return files[image_start:]
    return files[image_start:min(len(files), image_end + 1)]


def iter_images_recursive(root: Path) -> List[Path]:
    """Recursive reference discovery used by the FDT generator."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort(key=lambda p: _natural_key(str(p)))
    return files


# --------------------------------
# Image I/O
# --------------------------------

def b64_to_png(b64_str: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64_str))


# --------------------------------
# OpenAI Images
# --------------------------------

def call_images_edit_with_file(client: "OpenAI", model: str, image_path: Path, prompt: str):
    """Prefer images.edit if available; else fallback to images.edits for compatibility."""
    try:
        with open(image_path, "rb") as f:
            return client.images.edit(model=model, image=f, prompt=prompt, n=1)
    except AttributeError:
        with open(image_path, "rb") as f:
            return client.images.edits(model=model, image=f, prompt=prompt, n=1)


# --------------------------------
# Filesystem-safe filename helpers (FDT generator)
# --------------------------------

def safe_slug(text: str, max_len: int = 48) -> str:
    """
    Make a filesystem-safe, moderately short slug.
    - Normalize unicode
    - Replace non-alnum with '_'
    - Collapse repeats
    - If too long, keep head and add _md5 suffix
    """
    if not isinstance(text, str):
        text = str(text)
    norm = unicodedata.normalize("NFKC", text)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", norm).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if len(slug) <= max_len:
        return slug
    h = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
    head = slug[: max(8, max_len - 9)]
    return f"{head}_{h}"


def shorten_stem(stem: str, max_len: int = 160) -> str:
    """Ensure the final base filename stem isn't too long (before adding suffixes)."""
    if len(stem) <= max_len:
        return stem
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"{stem[: max_len-9]}_{h}"

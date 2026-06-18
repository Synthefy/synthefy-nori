#!/usr/bin/env python
"""Stack the small- and large-grid overview PNGs into one image (vertically).

Usage: uv run --no-sync --with matplotlib python benchmarks/combine_overviews.py
(Pillow ships with matplotlib.)
"""
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
TOP = HERE / "latency_overview.png"          # small grid
BOTTOM = HERE / "latency_large_overview.png"  # large grid
OUT = HERE / "latency_overview_combined.png"
GAP = 24  # white separator (px)


def main():
    a = Image.open(TOP).convert("RGB")
    b = Image.open(BOTTOM).convert("RGB")
    w = max(a.width, b.width)
    out = Image.new("RGB", (w, a.height + GAP + b.height), "white")
    out.paste(a, ((w - a.width) // 2, 0))
    out.paste(b, ((w - b.width) // 2, a.height + GAP))
    out.save(OUT)
    print(f"wrote {OUT.name}  ({out.width}x{out.height})")


if __name__ == "__main__":
    main()

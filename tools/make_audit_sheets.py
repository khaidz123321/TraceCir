#!/usr/bin/env python
"""Contact sheets of the reference images of compiled queries, for the manual
compiler audit (protocol Sec. 5.3). Runs next to the data (e.g. on the GPU box)
so only the small sheet JPGs need to be copied out.

Queries are taken in the order of the compiled JSONL; sheet k holds items
10(k-1)+1 .. 10k, each tile labelled with its 1-based number.

Example:
    python tools/make_audit_sheets.py --dataset cirr --data-root /workspace/data/CIRR \
        --compiled /workspace/outputs/cirr_compiled_q36.jsonl --out-dir /workspace/outputs/sheets_cirr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import CIRCODataset, CIRRDataset  # noqa: E402
from src.p0.run_e0 import _query_id  # noqa: E402

TILE_H = 260
PER_ROW = 5
PER_SHEET = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["circo", "cirr"], required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cls = CIRCODataset if args.dataset == "circo" else CIRRDataset
    items = cls(args.data_root, args.split, "relative", preprocess=lambda image: image)
    by_id = {_query_id(args.dataset, items[i]): i for i in range(len(items))}

    records = [json.loads(line) for line in open(args.compiled, encoding="utf-8")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = []
    for number, record in enumerate(records, start=1):
        image = items[by_id[record["query_id"]]]["reference_image"].convert("RGB")
        width = max(1, round(image.width * TILE_H / image.height))
        tile = image.resize((min(width, 420), TILE_H))
        ImageDraw.Draw(tile).rectangle((0, 0, 44, 24), fill="black")
        ImageDraw.Draw(tile).text((6, 5), str(number), fill="yellow")
        tiles.append(tile)

    for sheet_index in range(0, len(tiles), PER_SHEET):
        group = tiles[sheet_index : sheet_index + PER_SHEET]
        rows = [group[i : i + PER_ROW] for i in range(0, len(group), PER_ROW)]
        width = max(sum(t.width for t in row) + 6 * (len(row) - 1) for row in rows)
        sheet = Image.new("RGB", (width, len(rows) * (TILE_H + 6)), "white")
        for r, row in enumerate(rows):
            x = 0
            for tile in row:
                sheet.paste(tile, (x, r * (TILE_H + 6)))
                x += tile.width + 6
        sheet.save(out_dir / f"sheet_{sheet_index // PER_SHEET + 1:02d}.jpg", quality=85)
    print(f"{len(tiles)} tiles -> {len(range(0, len(tiles), PER_SHEET))} sheets in {out_dir}")


if __name__ == "__main__":
    main()

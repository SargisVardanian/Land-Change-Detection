#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the evaluated patchseg run}"
INPUT_DIR="${RUN_DIR}/evaluation/qcpr_mask_overlays"
OUTPUT="${INPUT_DIR}/contact_sheet.png"
PYTHON="${PYTHON:-python}"
test -d "$INPUT_DIR"

"$PYTHON" - "$INPUT_DIR" "$OUTPUT" <<'PY'
import math
import sys
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

root, output = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(path for path in root.glob("*.png") if path.name != output.name)
if not files:
    raise SystemExit(f"No QCPR overlays found in {root}")
thumb_w, thumb_h, columns = 480, 280, 3
rows = math.ceil(len(files) / columns)
sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 24)), "white")
draw = ImageDraw.Draw(sheet)
for index, path in enumerate(files):
    with Image.open(path) as image:
        tile = ImageOps.contain(image.convert("RGB"), (thumb_w, thumb_h))
    x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + 24)
    sheet.paste(tile, (x + (thumb_w - tile.width) // 2, y))
    draw.text((x + 4, y + thumb_h + 4), path.name, fill="black")
sheet.save(output)
print(output)
PY

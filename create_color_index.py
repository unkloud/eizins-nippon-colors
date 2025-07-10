#!/usr/bin/env python3
import json
from pathlib import Path

from eizins_nippon_colors import EizinsNipponColors, to_image, rgb_to_hex

all_colors = set()
all_pieces = EizinsNipponColors.enzin_nippon_colors(
    Path("./eizins_nippon_colors.jsonl")
)
for c in all_pieces:
    print(c.piece.title)
    img = to_image(c.piece.base64_encoded_img)
    img = img.convert("RGB")
    (width, height) = img.size
    for x in range(width):
        for y in range(height):
            pixel_rgb = img.getpixel((x, y))
            all_colors.add(rgb_to_hex(pixel_rgb))
print(len(all_colors))
with open("color_index.json", "w") as f:
    print(json.dumps(list(all_colors), indent=2, ensure_ascii=False), file=f)

#!/usr/bin/env python3
import json
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import colour

from nippon_colors import NipponColor


@cache
def hex_to_rgbi(hex: str):
    return int(hex[0:2], 16), int(hex[2:4], 16), int(hex[4:6], 16)


@cache
def rgb_to_lab(hex_color: str):
    rgb = colour.notation.HEX_to_RGB(hex_color)
    lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))
    return lab


def similarity_scores(
    hex_color: str, color_palettes: list[str], method="CIE 2000", threshold=15.0
):
    val = dict()
    for palette_color in color_palettes:
        palette_color_lab = rgb_to_lab(palette_color)
        hex_color_lab = rgb_to_lab(hex_color)
        delta_e = float(colour.delta_E(palette_color_lab, hex_color_lab, method=method))
        if delta_e <= threshold:
            val[palette_color] = delta_e
    return val


if __name__ == "__main__":
    nippon_colors = [c.hex_rgb for c in NipponColor.load(Path("./nippon_colors.json"))]
    with open("color_index.json") as f:
        hex_colors = json.load(f)
    with ProcessPoolExecutor() as pool:
        scores = pool.map(
            similarity_scores, hex_colors, [nippon_colors] * len(hex_colors)
        )
    if not Path("color_similarity.json").is_file():
        with open("color_similarity.json", "w") as f:
            print(json.dumps(dict(zip(hex_colors, scores)), indent=2), file=f)

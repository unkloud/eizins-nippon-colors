import base64
import colorsys
import json
import math
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Self

import numpy
from PIL import Image
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from tqdm import tqdm

from eizin_profolio import EizinWork
from nippon_colors import NipponColor


def patch_asscalar(a):
    return a.item()


setattr(numpy, "asscalar", patch_asscalar)


def load_json_file(filename):
    """Load and parse a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def to_image(base64_string: str) -> Image:
    """Decode base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def hex_to_rgb(hex_color: str) -> tuple[int, ...]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex string."""
    return "{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])


def color_distance(color1: str, color2: str) -> float:
    """
    Calculates the perceptual color difference between two RGB colors using the CIEDE2000 formula.
    This is a highly accurate, industry-standard metric for determining how different two colors
    appear to the human eye. The calculation is done in the perceptually uniform CIELAB color space.
    A lower Delta E (ΔE) value means the colors are more similar.
    - ΔE < 1.0: Difference is not perceptible by the human eye.
    - 1.0 < ΔE < 2.0: Perceptible only through close observation.
    - 2.0 < ΔE < 10.0: Perceptible at a glance.
    Args:
        color1: A tuple representing the first RGB color (R, G, B) from 0-255.
        color2: A tuple representing the second RGB color (R, G, B) from 0-255.
    Returns:
        The perceptual difference (Delta E 2000) between the two colors as a float.
    """
    # Create sRGBColor objects from the 0-255 integer tuples.
    # colormath requires RGB values to be normalized to a 0.0-1.0 scale.
    color1_rgb = sRGBColor.new_from_rgb_hex(color1)
    color2_rgb = sRGBColor.new_from_rgb_hex(color2)
    # Convert the sRGB color objects to the CIELAB color space.
    # The Delta E 2000 formula operates on colors in this space.
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)
    # Calculate the Delta E 2000 color difference.
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return float(delta_e)


def dominant_colors(image: Image, num_colors=10):
    """Extract dominant colors from an image using color quantization."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    quantized = image.quantize(colors=num_colors)
    palette = quantized.getpalette()
    colors = []
    for i in range(num_colors):
        r = palette[i * 3]
        g = palette[i * 3 + 1]
        b = palette[i * 3 + 2]
        colors.append((r, g, b))
    return colors


def significant_colors(
    image: Image.Image, colors: list[tuple[int, int, int]]
) -> list[str]:
    """
    Calculates the visual significance of dominant colors based on area, saliency, and position.
    Returns a list of tuples: [(significance_score, color), ...], sorted by significance.
    """
    # 1. Assign each pixel to the nearest dominant color
    pixel_map = image.load()
    color_assignments = {}  # Will store data for each dominant color
    for color in colors:
        color_assignments[color] = {"count": 0, "positions": []}
    # A simple way to map pixels to dominant colors (can be slow, better with numpy)
    for x in range(image.width):
        for y in range(image.height):
            pixel_rgb = pixel_map[x, y]
            # In a real scenario, use a proper color distance function here!
            # For simplicity, let's find the closest dominant color
            closest_color = min(
                colors, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, pixel_rgb))
            )

            color_assignments[closest_color]["count"] += 1
            color_assignments[closest_color]["positions"].append((x, y))
    # 2. Calculate scores for each factor
    total_pixels = image.width * image.height
    img_center = (image.width / 2, image.height / 2)
    max_dist = math.sqrt(img_center[0] ** 2 + img_center[1] ** 2)
    color_scores = []
    for color, data in color_assignments.items():
        if data["count"] == 0:
            continue
        # Area score
        proportion = data["count"] / total_pixels
        # Saliency score (using Saturation from HSL)
        # Normalize RGB from 0-255 to 0-1 for colorsys
        r, g, b = [v / 255.0 for v in color]
        _, _, saturation = colorsys.rgb_to_hls(r, g, b)
        # Position score
        if data["positions"]:
            avg_x = sum(p[0] for p in data["positions"]) / data["count"]
            avg_y = sum(p[1] for p in data["positions"]) / data["count"]
            dist_from_center = math.sqrt(
                (avg_x - img_center[0]) ** 2 + (avg_y - img_center[1]) ** 2
            )
            position_score = 1.0 - (dist_from_center / max_dist)
        else:
            position_score = 0
        # 3. Combine with weights
        w_area = 0.5
        w_saliency = 0.3
        w_position = 0.2
        final_score = (
            (w_area * proportion)
            + (w_saliency * saturation)
            + (w_position * position_score)
        )
        color_scores.append((final_score, rgb_to_hex(color)))
    # 4. Sort by final score
    color_scores.sort(key=lambda item: item[0], reverse=True)
    return color_scores


def find_closest_nippon_colors(
    target_rgb, nippon_colors: list[NipponColor], closest_n=5
) -> list[tuple[NipponColor, float]]:
    """Find the closest n nippon color to the target RGB color."""
    color_and_distance = []
    for color in nippon_colors:
        distance = color_distance(target_rgb, color.hex_rgb)
        color_and_distance.append((color, distance))
    color_and_distance.sort(key=lambda x: x[1])
    return color_and_distance[:closest_n]


@dataclass
class ColorWithSignificanceScore:
    hex_rgb_color: str
    significance_score: float
    nippon_colors_alternatives: list[tuple[NipponColor, float]]

    @classmethod
    def with_eizins_nippon_colors_alternatives(
        cls,
        hex_rgb_color: str,
        significance_score: float,
        nippon_colors: list[NipponColor],
    ):
        nippon_colors_alternatives = find_closest_nippon_colors(
            hex_rgb_color, nippon_colors
        )
        return ColorWithSignificanceScore(
            hex_rgb_color, significance_score, nippon_colors_alternatives
        )


@dataclass
class EizinsNipponColors:
    piece: EizinWork
    significant_colors: list[ColorWithSignificanceScore]

    @classmethod
    def from_eizin_work(cls, piece: EizinWork, nippon_colors: list[NipponColor]):
        print(f"Processing {piece.title}")
        image = to_image(piece.base64_encoded_img)
        visually_significant_colors = significant_colors(image, dominant_colors(image))
        return cls(
            piece,
            [
                ColorWithSignificanceScore.with_eizins_nippon_colors_alternatives(
                    hex_color, score, nippon_colors
                )
                for (score, hex_color) in visually_significant_colors
            ],
        )

    @classmethod
    def enzin_nippon_colors(cls, path: Path) -> list[Self]:
        enzins_colors = []
        for line in path.read_text().splitlines():
            line_obj = json.loads(line)
            work = EizinWork(**line_obj["piece"])
            enzins_colors.append(
                EizinsNipponColors(
                    piece=work,
                    significant_colors=[
                        ColorWithSignificanceScore(
                            hex_rgb_color=significant_color["hex_rgb_color"],
                            significance_score=significant_color["significance_score"],
                            nippon_colors_alternatives=[
                                (NipponColor(**color), similarity_score)
                                for (color, similarity_score) in significant_color[
                                    "nippon_colors_alternatives"
                                ]
                            ],
                        )
                        for significant_color in line_obj["significant_colors"]
                    ],
                )
            )
        return enzins_colors


def main():
    nippon_colors = NipponColor.load(Path("nippon_colors.json"))
    profolio = tqdm(
        [
            piece
            for piece in EizinWork.load_profolio(Path("eizin_profolio.json"))
            if piece.base64_encoded_img
        ]
    )
    eizins_nippon_colors = []
    for piece in profolio:
        eizins_nippon_colors.append(
            EizinsNipponColors.from_eizin_work(piece, nippon_colors)
        )
    with open("eizins_nippon_colors.jsonl", "w") as f:
        for enc in eizins_nippon_colors:
            print(json.dumps(asdict(enc), ensure_ascii=False), file=f, flush=True)


if __name__ == "__main__":
    main()

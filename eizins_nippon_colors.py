import base64
import colorsys
import json
import math
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy
from PIL import Image
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor

from eizin_profolio import EizinWork
from enc import NipponColor


def patch_asscalar(a):
    return a.item()


setattr(numpy, "asscalar", patch_asscalar)


def load_json_file(filename):
    """Load and parse a JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image."""
    # Remove data URL prefix if present
    if base64_string.startswith('data:'):
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex string."""
    return '{:02X}{:02X}{:02X}'.format(rgb[0], rgb[1], rgb[2])


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


def extract_dominant_colors(image: Image, num_colors=10):
    """Extract dominant colors from an image using color quantization."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    quantized = image.quantize(colors=num_colors)
    palette = quantized.getpalette()
    colors = []
    for i in range(num_colors):
        r = palette[i * 3]
        g = palette[i * 3 + 1]
        b = palette[i * 3 + 2]
        colors.append((r, g, b))
    return colors


def calculate_visual_significance(image: Image.Image, dominant_colors: list[tuple[int, int, int]]):
    """
    Calculates the visual significance of dominant colors based on area, saliency, and position.

    Returns a list of tuples: [(significance_score, color), ...], sorted by significance.
    """
    # 1. Assign each pixel to the nearest dominant color
    pixel_map = image.load()
    color_assignments = {}  # Will store data for each dominant color
    for color in dominant_colors:
        color_assignments[color] = {"count": 0, "positions": []}
    # A simple way to map pixels to dominant colors (can be slow, better with numpy)
    for x in range(image.width):
        for y in range(image.height):
            pixel_rgb = pixel_map[x, y]
            # In a real scenario, use a proper color distance function here!
            # For simplicity, let's find the closest dominant color
            closest_color = min(dominant_colors, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, pixel_rgb)))

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
            dist_from_center = math.sqrt((avg_x - img_center[0]) ** 2 + (avg_y - img_center[1]) ** 2)
            position_score = 1.0 - (dist_from_center / max_dist)
        else:
            position_score = 0
        # 3. Combine with weights
        w_area = 0.5
        w_saliency = 0.3
        w_position = 0.2
        final_score = (w_area * proportion) + (w_saliency * saturation) + (w_position * position_score)
        color_scores.append((final_score, color))
    # 4. Sort by final score
    color_scores.sort(key=lambda item: item[0], reverse=True)
    return color_scores


def find_closest_nippon_colors(target_rgb, nippon_colors: [NipponColor], closest_n=3) -> list[
    tuple[NipponColor, float]]:
    """Find the closest n nippon color to the target RGB color."""
    color_and_distance = []
    for color in nippon_colors:
        distance = color_distance(target_rgb, color.hex_rgb)
        color_and_distance.append((color, distance))
    color_and_distance.sort(key=lambda x: x[1])
    return color_and_distance[:closest_n]


@dataclass
class NipponColorSignificance:
    color: NipponColor
    score: float


@dataclass
class EizonsNipponColors:
    work: EizinWork
    nippon_color_palette: [NipponColorSignificance] = field(init=False)

    def __post_init__(self):
        pass


def main():
    nippon_colors = NipponColor.load(Path('nippon_colors.json'))
    profolio = EizinWork.load_profolio(Path('eizin_profolio.json'))
    for work in profolio:
        image = decode_base64_image(work.base64_encoded_img)
        dominant_colors = extract_dominant_colors(image)
        dominant_color_scores = calculate_visual_significance(image, dominant_colors)
        print("Extracted Color Palette:")
        for i, (score, color) in enumerate(dominant_color_scores, 1):
            hex_color = rgb_to_hex(color)
            print(f"{i:2d}. RGB: {color} | Hex: #{hex_color}, sigificance: {score:.2f}")
        for i, (score, color) in enumerate(dominant_color_scores, 1):
            hex_color = rgb_to_hex(color)
            colors = find_closest_nippon_colors(hex_color, nippon_colors)
            for closest_nippon, distance in colors:
                print(f"{i:<3} RGB: {color} #{hex_color:<10} | "
                      f"{closest_nippon.english_name:<15} (#{closest_nippon.hex_rgb}) | "
                      f"{distance:.2f}")
                print(f"    {'':20} | {closest_nippon.kanji_name}")


if __name__ == "__main__":
    main()

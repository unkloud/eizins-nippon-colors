#!/usr/bin/env python3
import base64
import json
from io import BytesIO
from pathlib import Path
from random import choice

import msgpack
import numpy as np
from PIL import Image, ImageFilter

from eizins_nippon_colors import EizinsNipponColors, EizinWork, rgb_to_hex, hex_to_rgb


def to_html(piece: EizinWork) -> str:
    return f"""<article>
    <header>{piece.title}</header>
    <img src="data:image/png;base64,{piece.base64_encoded_img}">
    <p>{piece.desc}</p>
</article>"""


def to_file(piece: EizinWork, file_name: str):
    with open(file_name, "wb") as f:
        f.write(base64.b64decode(piece.base64_encoded_img.encode()))


def enzin_in_nippon_colors_improved(
    piece: EizinWork, color_mapping: dict[str, list[str]]
) -> EizinWork:
    img = Image.open(BytesIO(base64.b64decode(piece.base64_encoded_img))).convert("RGB")
    img.save(f"{piece.title}.png")
    # Apply Floyd-Steinberg dithering
    img_array = np.array(img, dtype=np.float32)
    height, width = img_array.shape[:2]
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            old_rgb = tuple(old_pixel.astype(int))
            rgb_hex = rgb_to_hex(old_rgb)
            if alternative_nippon_colors := color_mapping.get(rgb_hex):
                # Get the best color (lowest score = most similar)
                color_and_scores = sorted(
                    alternative_nippon_colors.items(), key=lambda x: x[1]
                )
                new_hex = color_and_scores[0][0]
                new_pixel = np.array(hex_to_rgb(new_hex), dtype=np.float32)
            else:
                new_pixel = old_pixel
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            # Distribute quantization error (Floyd-Steinberg)
            if x + 1 < width:
                img_array[y, x + 1] += quant_error * 7 / 16
            if y + 1 < height:
                if x - 1 >= 0:
                    img_array[y + 1, x - 1] += quant_error * 3 / 16
                img_array[y + 1, x] += quant_error * 5 / 16
                if x + 1 < width:
                    img_array[y + 1, x + 1] += quant_error * 1 / 16
    # Convert back to PIL Image
    dithered_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    dithered_img.save(f"{piece.title}_dithered.png")
    return EizinWork(
        piece.title, piece.desc, base64.b64encode(dithered_img.tobytes()).decode()
    )


def enzin_in_nippon_colors_enhanced(
    piece: EizinWork, color_mapping: dict[str, list[str]]
) -> EizinWork:
    img = Image.open(BytesIO(base64.b64decode(piece.base64_encoded_img))).convert("RGB")
    img.save(f"{piece.title}.png")
    # Pre-processing: Light blur to reduce harsh edges
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    # Detect edges for adaptive dithering
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges.convert("L"))
    # Apply enhanced dithering
    img_array = np.array(img, dtype=np.float32)
    height, width = img_array.shape[:2]
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            old_rgb = tuple(old_pixel.astype(int))
            rgb_hex = rgb_to_hex(old_rgb)
            if alternative_nippon_colors := color_mapping.get(rgb_hex):
                color_and_scores = sorted(
                    alternative_nippon_colors.items(), key=lambda x: x[1]
                )
                # Add some randomness to avoid banding
                if len(color_and_scores) > 1 and np.random.random() < 0.1:
                    new_hex = color_and_scores[1][0]  # Second best sometimes
                else:
                    new_hex = color_and_scores[0][0]
                new_pixel = np.array(hex_to_rgb(new_hex), dtype=np.float32)
            else:
                new_pixel = old_pixel
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            # Adaptive dithering strength
            edge_strength = edge_array[y, x] / 255.0
            dither_strength = 0.4 + (edge_strength * 0.4)  # 0.4 to 0.8
            # Distribute error with adaptive strength
            if x + 1 < width:
                img_array[y, x + 1] += quant_error * (7 / 16) * dither_strength
            if y + 1 < height:
                if x - 1 >= 0:
                    img_array[y + 1, x - 1] += quant_error * (3 / 16) * dither_strength
                img_array[y + 1, x] += quant_error * (5 / 16) * dither_strength
                if x + 1 < width:
                    img_array[y + 1, x + 1] += quant_error * (1 / 16) * dither_strength

    # Post-processing: Very light blur to smooth dithering artifacts
    dithered_img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    dithered_img = dithered_img.filter(ImageFilter.GaussianBlur(radius=0.2))
    dithered_img.save(f"{piece.title}_enhanced.png")
    return EizinWork(
        piece.title, piece.desc, base64.b64encode(dithered_img.tobytes()).decode()
    )


def enzin_in_nippon_colors(
    piece: EizinWork, color_mapping: dict[str, list[str]]
) -> EizinWork:
    img = Image.open(BytesIO(base64.b64decode(piece.base64_encoded_img))).convert("RGB")
    img.save(f"{piece.title}.png")
    for i in range(img.width):
        for j in range(img.height):
            pixel_rgb = img.getpixel((i, j))
            rgb_hex = rgb_to_hex(pixel_rgb)
            if alternative_nippon_colors := color_mapping.get(rgb_hex):
                color_and_scores = sorted(
                    [
                        (color, score)
                        for color, score in alternative_nippon_colors.items()
                    ],
                    key=lambda x: x[1],
                )
                rgb_hex = color_and_scores[0][0]
                img.putpixel((i, j), hex_to_rgb(rgb_hex))
    img.save(f"{piece.title}_new.png")
    return EizinWork(piece.title, piece.desc, base64.b64encode(img.tobytes()).decode())


enzins_colors = EizinsNipponColors.enzin_nippon_colors(
    Path("eizins_nippon_colors.jsonl")
)
if not (compressed_colors := Path("color_similarity.json.msgpack")).is_file():
    color_replacement = json.load(open("color_similarity.json"))
    compressed_colors.write_bytes(msgpack.packb(color_replacement))
alternative_colors = msgpack.unpackb(compressed_colors.read_bytes())
for chosen_piece in enzins_colors:
    print(f"{chosen_piece.piece.title=}")
    print(f"{len(alternative_colors)=}")
    new_piece = enzin_in_nippon_colors_enhanced(chosen_piece.piece, alternative_colors)
    # print(new_piece.base64_encoded_img)
    to_file(new_piece, f"new_piece.png")

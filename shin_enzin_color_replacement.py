#!/usr/bin/env python3
"""
Direct Color Replacement Pipeline: Eizin → Nippon Colors
Uses existing color similarity mapping for precise color replacement
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import msgpack
import numpy as np
from PIL import Image, ImageFilter

from eizin_profolio import EizinWork
from eizins_nippon_colors import rgb_to_hex, hex_to_rgb


class DirectColorReplacer:
    def __init__(self):
        # Load your existing color similarity data
        if Path("color_similarity.json.msgpack").exists():
            self.color_mapping = msgpack.unpackb(
                Path("color_similarity.json.msgpack").read_bytes()
            )
        else:
            with open("color_similarity.json") as f:
                self.color_mapping = json.load(f)

    def replace_colors(self, eizin_work: EizinWork) -> Image.Image:
        """Replace all colors with closest Nippon colors"""

        # Load original image
        image_data = base64.b64decode(eizin_work.base64_encoded_img)
        img = Image.open(BytesIO(image_data)).convert("RGB")

        # Convert to numpy for faster processing
        img_array = np.array(img)
        height, width = img_array.shape[:2]

        # Create output array
        output_array = img_array.copy()

        # Replace each pixel
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x]
                original_hex = rgb_to_hex((r, g, b))

                # Find closest Nippon color
                if nippon_alternatives := self.color_mapping.get(original_hex):
                    # Get the best match (lowest similarity score)
                    best_match = min(nippon_alternatives.items(), key=lambda x: x[1])
                    new_hex = best_match[0]
                    new_rgb = hex_to_rgb(new_hex)
                    output_array[y, x] = new_rgb

        return Image.fromarray(output_array)

    def replace_colors_with_dithering(self, eizin_work: EizinWork) -> Image.Image:
        """Replace colors with Floyd-Steinberg dithering for smoother transitions"""

        # Load original image
        image_data = base64.b64decode(eizin_work.base64_encoded_img)
        img = Image.open(BytesIO(image_data)).convert("RGB")

        # Apply slight blur to reduce harsh edges
        img = img.filter(ImageFilter.GaussianBlur(radius=0.2))

        # Convert to float array for dithering
        img_array = np.array(img, dtype=np.float32)
        height, width = img_array.shape[:2]

        # Apply Floyd-Steinberg dithering
        for y in range(height):
            for x in range(width):
                old_pixel = img_array[y, x]
                old_rgb = tuple(old_pixel.astype(int))
                original_hex = rgb_to_hex(old_rgb)

                # Find closest Nippon color
                if nippon_alternatives := self.color_mapping.get(original_hex):
                    best_match = min(nippon_alternatives.items(), key=lambda x: x[1])
                    new_hex = best_match[0]
                    new_pixel = np.array(hex_to_rgb(new_hex), dtype=np.float32)
                else:
                    new_pixel = old_pixel

                # Set new pixel
                img_array[y, x] = new_pixel

                # Calculate quantization error
                quant_error = old_pixel - new_pixel

                # Distribute error to neighboring pixels (Floyd-Steinberg)
                if x + 1 < width:
                    img_array[y, x + 1] += quant_error * 7 / 16
                if y + 1 < height:
                    if x - 1 >= 0:
                        img_array[y + 1, x - 1] += quant_error * 3 / 16
                    img_array[y + 1, x] += quant_error * 5 / 16
                    if x + 1 < width:
                        img_array[y + 1, x + 1] += quant_error * 1 / 16

        # Convert back to PIL Image
        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

        # Optional: slight blur to smooth dithering artifacts
        result = result.filter(ImageFilter.GaussianBlur(radius=0.1))

        return result


def main():
    # Initialize color replacer
    replacer = DirectColorReplacer()

    # Load artworks
    eizin_works = EizinWork.load_profolio(Path("eizin_profolio.json"))

    # Create output directory
    output_dir = Path("nippon_colored")
    output_dir.mkdir(exist_ok=True)

    # Process each artwork
    for i, work in enumerate(eizin_works):
        if not work.base64_encoded_img:
            continue

        print(f"Processing: {work.title}")

        try:
            # Method 1: Direct replacement
            direct_result = replacer.replace_colors(work)

            # Method 2: With dithering (smoother)
            dithered_result = replacer.replace_colors_with_dithering(work)

            # Save both versions
            safe_title = "".join(
                c for c in work.title if c.isalnum() or c in (" ", "_")
            ).strip()
            safe_title = safe_title.replace(" ", "_")

            direct_result.save(output_dir / f"{i:03d}_{safe_title}_direct.png")
            dithered_result.save(output_dir / f"{i:03d}_{safe_title}_dithered.png")

            print(f"  ✓ Saved both versions")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print("Processing complete!")


if __name__ == "__main__":
    main()

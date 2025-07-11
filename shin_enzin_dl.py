#!/usr/bin/env python3
# !/usr/bin/env python3
import base64
from io import BytesIO
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from eizin_profolio import EizinWork


class SimpleGenerativePipeline:
    def __init__(self):
        # Load Img2Img pipeline - this is all we need!
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")
        # Memory optimization for 4060 Ti
        # self.pipe.enable_memory_efficient_attention()
        self.pipe.enable_xformers_memory_efficient_attention()

    def generate(self, eizin_work: EizinWork, strength: float = 0.7) -> Image.Image:
        # Load original image
        image_data = base64.b64decode(eizin_work.base64_encoded_img)
        original_image = Image.open(BytesIO(image_data)).convert("RGB")
        # Resize to standard size
        original_image = original_image.resize((512, 512), Image.Resampling.LANCZOS)
        # Generate with minimal prompt
        result = self.pipe(
            prompt="traditional Japanese painting with natural colors",
            image=original_image,
            strength=strength,  # How much to change (0.1 = subtle, 0.9 = dramatic)
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
        return result


def main():
    pipeline = SimpleGenerativePipeline()
    eizin_works = EizinWork.load_profolio(Path("eizin_profolio.json"))
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)
    for i, work in enumerate(eizin_works[:5]):  # Process first 5
        if not work.base64_encoded_img:
            continue
        print(f"Processing: {work.title}")
        # Generate with different strengths
        for strength in [0.3, 0.5, 0.7]:
            try:
                result = pipeline.generate(work, strength=strength)
                # Save result
                filename = f"{i:03d}_{strength:.1f}_{work.title.replace(' ', '_')}.png"
                result.save(output_dir / filename)
                print(f"  Saved: {filename}")
            except Exception as e:
                print(f"  Error: {e}")
                continue


if __name__ == "__main__":
    main()

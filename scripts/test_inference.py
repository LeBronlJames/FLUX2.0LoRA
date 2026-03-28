#!/usr/bin/env python3
"""
test_inference.py — Art-to-Photorealistic Inference Pipeline
FLUX.2 Klein 9B + Consistency LoRA

支持三种推理模式:
  - art2real   : 艺术风格图 → 写实照片
  - inpaint    : 局部 Mask 区域写实替换（脸部/身体）
  - outpaint   : 无限画布扩展（叙事系列生成）

用法:
    # 艺术转写实
    python test_inference.py --mode art2real \
        --input_image ../examples/input_art/sample_anime.png \
        --lora_path ../checkpoints/consistency_lora.safetensors \
        --output_path ../examples/output_real/result.png

    # Inpaint 脸部替换
    python test_inference.py --mode inpaint \
        --input_image ../examples/input_art/sample.png \
        --mask_image ../examples/input_art/mask_face.png \
        --lora_path ../checkpoints/consistency_lora.safetensors \
        --output_path ../examples/output_real/inpainted.png

    # 叙事系列生成（批量 outpaint）
    python test_inference.py --mode narrative \
        --input_image ../examples/output_real/scene_01.png \
        --prompts "desert starry night, lying on car roof" \
                  "tropical beach at sunset, lounger by the sea" \
        --lora_path ../checkpoints/consistency_lora.safetensors \
        --output_dir ../examples/narrative_series/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_LORA_WEIGHT = 0.9
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_NUM_STEPS = 28
DEFAULT_SEED = 42

ART2REAL_POSITIVE = (
    "MYCHAR photorealistic portrait, {prompt}, "
    "cinematic lighting, 8k ultra-detailed, sharp focus, "
    "professional photography, consistent identity"
)
ART2REAL_NEGATIVE = (
    "anime, cartoon, painting, illustration, sketch, "
    "low quality, blurry, deformed anatomy, watermark"
)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------
class Art2RealPipeline:
    """
    High-level wrapper around FLUX.2 diffusion pipeline with Consistency LoRA.

    Parameters
    ----------
    model_path : str
        HuggingFace repo ID or local path to the FLUX.2 base model.
    lora_path : str | None
        Path to the Consistency LoRA .safetensors checkpoint.
        If None, runs without LoRA (baseline comparison).
    lora_weight : float
        LoRA blending weight in [0, 1]. Higher = stronger identity control.
    device : str
        Torch device string, e.g. "cuda" or "cpu".
    dtype : torch.dtype
        Inference dtype. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        lora_path: Optional[str] = None,
        lora_weight: float = DEFAULT_LORA_WEIGHT,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_weight = lora_weight
        self.device = device
        self.dtype = dtype
        self._pipe = None
        self._inpaint_pipe = None

    # ------------------------------------------------------------------
    # Lazy loading helpers
    # ------------------------------------------------------------------
    def _load_pipeline(self) -> None:
        """Load the base FLUX.2 image-to-image pipeline (lazy)."""
        if self._pipe is not None:
            return
        try:
            from diffusers import FluxImg2ImgPipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "diffusers >= 0.27.0 is required. "
                "Run `pip install -r requirements.txt`."
            ) from exc

        logger.info("Loading base model from %s …", self.model_path)
        self._pipe = FluxImg2ImgPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        ).to(self.device)

        if self.lora_path and os.path.isfile(self.lora_path):
            logger.info("Loading Consistency LoRA from %s (weight=%.2f) …", self.lora_path, self.lora_weight)
            self._pipe.load_lora_weights(self.lora_path)
            self._pipe.fuse_lora(lora_scale=self.lora_weight)
        elif self.lora_path:
            logger.warning("LoRA path not found: %s — running without LoRA.", self.lora_path)

    def _load_inpaint_pipeline(self) -> None:
        """Load the FLUX.2 inpainting pipeline (lazy)."""
        if self._inpaint_pipe is not None:
            return
        try:
            from diffusers import FluxInpaintPipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "diffusers >= 0.27.0 with FluxInpaintPipeline is required."
            ) from exc

        logger.info("Loading inpaint model from %s …", self.model_path)
        self._inpaint_pipe = FluxInpaintPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        ).to(self.device)

        if self.lora_path and os.path.isfile(self.lora_path):
            self._inpaint_pipe.load_lora_weights(self.lora_path)
            self._inpaint_pipe.fuse_lora(lora_scale=self.lora_weight)

    # ------------------------------------------------------------------
    # Core generation methods
    # ------------------------------------------------------------------
    def generate(
        self,
        input_image: str | Image.Image,
        prompt: str,
        negative_prompt: str = ART2REAL_NEGATIVE,
        mode: str = "art2real",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        strength: float = 0.75,
        seed: int = DEFAULT_SEED,
    ) -> Image.Image:
        """
        Generate a photorealistic image from an artistic input.

        Parameters
        ----------
        input_image : str or PIL.Image
            Path to or loaded artistic input image.
        prompt : str
            Text description of the desired photorealistic output.
        mode : str
            One of "art2real" (img2img) or "outpaint" (canvas extension).
        strength : float
            Denoising strength for img2img (0.0 = no change, 1.0 = full redraw).
        """
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        input_image = input_image.resize((width, height))

        full_prompt = ART2REAL_POSITIVE.format(prompt=prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        self._load_pipeline()

        logger.info("Generating [mode=%s] | steps=%d | guidance=%.1f | strength=%.2f",
                    mode, num_inference_steps, guidance_scale, strength)

        t0 = time.time()
        result = self._pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
            width=width,
            height=height,
        ).images[0]
        elapsed = time.time() - t0
        logger.info("Generation complete in %.1f s", elapsed)
        return result

    def inpaint(
        self,
        input_image: str | Image.Image,
        mask_image: str | Image.Image,
        prompt: str,
        negative_prompt: str = ART2REAL_NEGATIVE,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        strength: float = 0.85,
        seed: int = DEFAULT_SEED,
    ) -> Image.Image:
        """
        Inpaint the masked region with a photorealistic identity-consistent output.

        The mask should be a grayscale image where white (255) = region to
        replace, black (0) = region to preserve.
        """
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        if isinstance(mask_image, str):
            mask_image = Image.open(mask_image).convert("L")

        input_image = input_image.resize((width, height))
        mask_image = mask_image.resize((width, height))

        full_prompt = ART2REAL_POSITIVE.format(prompt=prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        self._load_inpaint_pipeline()

        logger.info("Inpainting | steps=%d | guidance=%.1f | strength=%.2f",
                    num_inference_steps, guidance_scale, strength)

        t0 = time.time()
        result = self._inpaint_pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
            width=width,
            height=height,
        ).images[0]
        elapsed = time.time() - t0
        logger.info("Inpaint complete in %.1f s", elapsed)
        return result

    def generate_narrative_series(
        self,
        base_image: str | Image.Image,
        scene_prompts: List[str],
        output_dir: str,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate a multi-scene narrative series from a base image.

        Each scene builds on the previous output, maintaining character
        identity via the Consistency LoRA while adapting to new scene prompts.

        Parameters
        ----------
        base_image : str or PIL.Image
            Starting image (e.g. the base art2real conversion result).
        scene_prompts : list of str
            Ordered list of scene descriptions for consecutive narrative frames.
        output_dir : str
            Directory to save the generated series.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        current_image = base_image

        for idx, scene_prompt in enumerate(scene_prompts, start=1):
            logger.info("Generating narrative scene %d/%d: %s", idx, len(scene_prompts), scene_prompt[:60])
            scene_result = self.generate(
                input_image=current_image,
                prompt=scene_prompt,
                seed=seed + idx,
                strength=0.65,
                **kwargs,
            )
            out_path = os.path.join(output_dir, f"scene_{idx:02d}.png")
            scene_result.save(out_path)
            logger.info("Saved scene %d → %s", idx, out_path)
            results.append(scene_result)
            current_image = scene_result  # next scene uses current output as base

        logger.info("Narrative series complete — %d scenes saved to %s", len(results), output_dir)
        return results


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLUX.2 Art-to-Real Inference")
    parser.add_argument(
        "--mode",
        choices=["art2real", "inpaint", "narrative"],
        default="art2real",
        help="Inference mode.",
    )
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--mask_image", type=str, default=None, help="Path to mask image (inpaint mode).")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["photorealistic portrait, cinematic lighting, 8k ultra-detailed"],
        help="Text prompt(s). Use multiple values for narrative mode.",
    )
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA .safetensors checkpoint.")
    parser.add_argument("--lora_weight", type=float, default=DEFAULT_LORA_WEIGHT, help="LoRA weight [0-1].")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL, help="Base model path.")
    parser.add_argument("--output_path", type=str, default="./output.png", help="Output image path (art2real/inpaint).")
    parser.add_argument("--output_dir", type=str, default="./narrative_series/", help="Output dir (narrative mode).")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    pipeline = Art2RealPipeline(
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_weight=args.lora_weight,
        device=args.device,
    )

    if args.mode == "art2real":
        result = pipeline.generate(
            input_image=args.input_image,
            prompt=args.prompts[0],
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            seed=args.seed,
        )
        out_path = args.output_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        logger.info("Saved result → %s", out_path)

    elif args.mode == "inpaint":
        if not args.mask_image:
            logger.error("--mask_image is required for inpaint mode.")
            sys.exit(1)
        result = pipeline.inpaint(
            input_image=args.input_image,
            mask_image=args.mask_image,
            prompt=args.prompts[0],
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            seed=args.seed,
        )
        out_path = args.output_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        logger.info("Saved inpaint result → %s", out_path)

    elif args.mode == "narrative":
        pipeline.generate_narrative_series(
            base_image=args.input_image,
            scene_prompts=args.prompts,
            output_dir=args.output_dir,
            seed=args.seed,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )


if __name__ == "__main__":
    main()

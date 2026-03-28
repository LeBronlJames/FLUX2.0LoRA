"""
test_inference.py — Art-to-Real inference script using FLUX.2 + Consistency LoRA

Usage:
    # Text-to-image (art2real)
    python scripts/test_inference.py \\
        --mode t2i \\
        --prompt "[trigger] photorealistic portrait, 8k" \\
        --lora path/to/flux_consistency_lora.safetensors \\
        --output examples/output_real/photorealistic_output.png

    # Image-to-image inpaint (identity migration)
    python scripts/test_inference.py \\
        --mode inpaint \\
        --input  examples/input_art/anime_portrait_input.png \\
        --mask   examples/input_art/inpaint_mask.png \\
        --prompt "[trigger] photorealistic face, ultra detailed, 8k" \\
        --lora   path/to/flux_consistency_lora.safetensors \\
        --output examples/output_real/inpaint_output.png

Requirements:
    pip install diffusers>=0.30.0 transformers>=4.43.0 accelerate safetensors Pillow torch
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image


def load_pipeline(mode: str, lora_path: str | None, lora_weight: float, device: str):
    """Load a FLUX pipeline for the requested mode, optionally with a LoRA."""
    try:
        from diffusers import FluxImg2ImgPipeline, FluxInpaintPipeline, FluxPipeline
    except ImportError as exc:
        print(
            "ERROR: diffusers not installed. Run: pip install diffusers>=0.30.0",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    model_id = "black-forest-labs/FLUX.1-schnell"
    dtype = torch.bfloat16

    print(f"[inference] Loading FLUX.2 klein ({mode}) from {model_id} ...")

    if mode == "t2i":
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    elif mode == "img2img":
        pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    elif mode == "inpaint":
        pipe = FluxInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose from t2i / img2img / inpaint.")

    pipe.to(device)

    if lora_path:
        if not os.path.isfile(lora_path):
            print(f"WARNING: LoRA file not found: {lora_path}", file=sys.stderr)
        else:
            print(f"[inference] Loading LoRA weights from {lora_path} (weight={lora_weight})")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_weight)

    return pipe


def run_t2i(pipe, args) -> Image.Image:
    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return result.images[0]


def run_img2img(pipe, args) -> Image.Image:
    input_image = Image.open(args.input).convert("RGB").resize((args.width, args.height))
    result = pipe(
        prompt=args.prompt,
        image=input_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return result.images[0]


def run_inpaint(pipe, args) -> Image.Image:
    if not args.mask:
        raise ValueError("--mask is required for inpaint mode")
    input_image = Image.open(args.input).convert("RGB").resize((args.width, args.height))
    mask_image = Image.open(args.mask).convert("L").resize((args.width, args.height))
    result = pipe(
        prompt=args.prompt,
        image=input_image,
        mask_image=mask_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return result.images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUX.2 Consistency LoRA inference — Art-to-Real conversion"
    )
    parser.add_argument(
        "--mode",
        choices=["t2i", "img2img", "inpaint"],
        default="t2i",
        help="Inference mode",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt (include trigger word)")
    parser.add_argument("--input", default=None, help="Input image path (img2img / inpaint)")
    parser.add_argument("--mask", default=None, help="Mask image path (inpaint only)")
    parser.add_argument(
        "--lora", default=None, help="Path to .safetensors LoRA file"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="LoRA fusion scale (0.0–1.0)"
    )
    parser.add_argument(
        "--output",
        default="output.png",
        help="Output image save path",
    )
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--strength", type=float, default=0.85, help="Denoising strength (img2img/inpaint)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[inference] Mode={args.mode} | Device={args.device} | Seed={args.seed}")
    pipe = load_pipeline(args.mode, args.lora, args.lora_weight, args.device)

    if args.mode == "t2i":
        image = run_t2i(pipe, args)
    elif args.mode == "img2img":
        image = run_img2img(pipe, args)
    else:
        image = run_inpaint(pipe, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))
    print(f"[inference] Saved output to: {output_path}")


if __name__ == "__main__":
    main()

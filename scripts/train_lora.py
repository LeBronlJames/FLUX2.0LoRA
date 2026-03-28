#!/usr/bin/env python3
"""
train_lora.py — FLUX.2 Klein 9B Consistency LoRA Training Script
使用 Ostris AI-Toolkit 在 24GB 显存下训练人物一致性 LoRA

依赖:
    pip install -r requirements.txt
    git clone https://github.com/ostris/ai-toolkit.git

用法:
    python train_lora.py --config_path ../config/consistency_lora_24gb.yaml \
                         --output_dir ../checkpoints/ \
                         --logging_dir ../logs/
"""

import argparse
import os
import sys
import logging
import yaml
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default training configuration (mirrors docs/training_config.md)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "job": "extension",
    "config": {
        "name": "flux2_consistency_lora_v1",
        "process": [
            {
                "type": "sd_trainer",
                "device": "cuda:0",
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-dev",
                    "is_flux": True,
                    "quantize": True,
                },
                "network": {
                    "type": "lora",
                    "linear": 16,
                    "linear_alpha": 16,
                },
                "train": {
                    "batch_size": 1,
                    "steps": 2500,
                    "gradient_accumulation_steps": 4,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "optimizer": "adamw8bit",
                    "learning_rate": 5e-5,
                    "lr_scheduler": "cosine",
                    "lr_warmup_steps": 100,
                    "noise_scheduler": "flow_match",
                    "timestep_sampling": "sigmoid",
                    "dtype": "bf16",
                    "train_dtype": "fp8",
                    "max_grad_norm": 1.0,
                    "gradient_checkpointing": True,
                },
                "datasets": [
                    {
                        "folder_path": "./training_data/art_real_pairs",
                        "caption_ext": "txt",
                        "caption_dropout_rate": 0.05,
                        "shuffle_tokens": False,
                        "cache_latents_to_disk": True,
                        "resolution": [[512, 512], [768, 512], [512, 768]],
                    }
                ],
                "save": {
                    "dtype": "float16",
                    "save_every": 500,
                    "max_step_saves_to_keep": 4,
                    "push_to_hub": False,
                    "output_dir": "./checkpoints/",
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": 500,
                    "width": 1024,
                    "height": 1024,
                    "prompts": [
                        "photorealistic portrait of a woman, cinematic lighting, 8k, ultra-detailed",
                        "photorealistic full body shot, forest background, natural lighting",
                    ],
                    "neg": "anime, cartoon, painting, illustration, low quality, blurry",
                    "seed": 42,
                    "walk_seed": True,
                    "guidance_scale": 3.5,
                    "sample_steps": 28,
                },
            }
        ],
    },
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Consistency LoRA for FLUX.2 Klein 9B"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to YAML training config. If not provided, uses built-in defaults.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/",
        help="Directory to save LoRA checkpoints.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs/",
        help="Directory to save training logs.",
    )
    parser.add_argument(
        "--training_data_dir",
        type=str,
        default="./training_data/art_real_pairs",
        help="Directory containing art-real paired training images and captions.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Base model path or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5).",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=2500,
        help="Total training steps (default: 2500).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per device (default: 1).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from, or 'latest' to auto-detect.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


def merge_args_into_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments where provided."""
    process = config["config"]["process"][0]
    train = process["train"]
    network = process["network"]

    # Override model
    process["model"]["name_or_path"] = args.model_name_or_path

    # Override network
    network["linear"] = args.lora_rank
    network["linear_alpha"] = args.lora_alpha

    # Override training params
    train["batch_size"] = args.batch_size
    train["steps"] = args.train_steps
    train["learning_rate"] = args.learning_rate

    # Override data path
    if process.get("datasets"):
        process["datasets"][0]["folder_path"] = args.training_data_dir

    # Override save dir
    process["save"]["output_dir"] = args.output_dir

    # Set run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["config"]["name"] = f"flux2_consistency_lora_{timestamp}"

    return config


def save_effective_config(config: dict, output_dir: str) -> str:
    """Save the effective training config for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, "effective_config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info("Effective config saved to %s", config_save_path)
    return config_save_path


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def check_environment() -> None:
    """Check that required packages and CUDA are available."""
    import importlib

    required = ["torch", "diffusers", "transformers", "accelerate"]
    missing = []
    for pkg in required:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        raise RuntimeError(
            f"Missing required packages: {missing}. "
            "Run `pip install -r requirements.txt` first."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This training script requires a GPU with "
            "at least 24GB VRAM."
        )

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(
        "GPU: %s | VRAM: %.1f GB",
        torch.cuda.get_device_name(0),
        vram_gb,
    )
    if vram_gb < 20:
        logger.warning(
            "Detected %.1f GB VRAM. Recommended minimum is 24 GB. "
            "Training may fail due to OOM.",
            vram_gb,
        )


def validate_training_data(data_dir: str) -> int:
    """Check that training data directory contains paired images and captions."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {data_dir}. "
            "Please prepare paired art/real images and captions."
        )

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = [
        f for f in data_path.iterdir() if f.suffix.lower() in image_extensions
    ]
    captions = [f for f in data_path.iterdir() if f.suffix.lower() == ".txt"]

    logger.info(
        "Training data: %d images, %d captions found in %s",
        len(images),
        len(captions),
        data_dir,
    )

    if len(images) == 0:
        raise ValueError(
            f"No training images found in {data_dir}. "
            "Expected image files (.png/.jpg/.jpeg/.webp)."
        )
    if len(captions) == 0:
        logger.warning(
            "No caption .txt files found. "
            "Training without captions may reduce quality."
        )

    return len(images)


def run_training(config: dict, args: argparse.Namespace) -> None:
    """
    Launch training via Ostris ai-toolkit.

    The ai-toolkit must be installed or present on PYTHONPATH.
    See: https://github.com/ostris/ai-toolkit
    """
    try:
        # ai-toolkit entry point
        from toolkit.job import get_job  # type: ignore

        job = get_job(config, config["config"]["name"])
        job.run()
        job.cleanup()
        logger.info("Training completed successfully.")
    except ImportError:
        logger.error(
            "Could not import 'toolkit.job' from ai-toolkit. "
            "Please install Ostris ai-toolkit:\n"
            "  git clone https://github.com/ostris/ai-toolkit.git\n"
            "  cd ai-toolkit && pip install -e ."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FLUX.2 Klein 9B — Consistency LoRA Training")
    logger.info("=" * 60)

    # 1. Environment check
    check_environment()

    # 2. Load / build config
    if args.config_path and os.path.isfile(args.config_path):
        config = load_config(args.config_path)
    else:
        logger.info("No config file specified. Using built-in defaults.")
        config = DEFAULT_CONFIG.copy()

    config = merge_args_into_config(config, args)

    # 3. Validate training data
    validate_training_data(args.training_data_dir)

    # 4. Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    # 5. Save effective config for reproducibility
    save_effective_config(config, args.output_dir)

    # 6. Log key hyperparameters
    process = config["config"]["process"][0]
    train = process["train"]
    network = process["network"]
    logger.info("Model       : %s", process["model"]["name_or_path"])
    logger.info("LoRA rank   : %d / alpha: %d", network["linear"], network["linear_alpha"])
    logger.info("LR          : %g", train["learning_rate"])
    logger.info("Steps       : %d", train["steps"])
    logger.info("Batch size  : %d (grad_accum × %d)", train["batch_size"], train["gradient_accumulation_steps"])
    logger.info("Output dir  : %s", args.output_dir)

    # 7. Launch training
    logger.info("Starting training...")
    run_training(config, args)


if __name__ == "__main__":
    main()

"""
train_lora.py — Ostris AI-Toolkit configuration launcher for FLUX.2 Consistency LoRA

Usage:
    python scripts/train_lora.py [--config CONFIG_PATH] [--output_dir OUTPUT_DIR]

This script generates a ready-to-use YAML config for the Ostris AI-Toolkit
and (optionally) launches training directly via subprocess.

Requirements:
    pip install ai-toolkit  # or clone https://github.com/ostris/ai-toolkit
"""

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULTS = {
    "model_name": "black-forest-labs/FLUX.1-schnell",
    "lora_rank": 16,
    "lora_alpha": 16,
    "learning_rate": 5e-5,
    "lr_scheduler": "cosine",
    "warmup_steps": 100,
    "max_train_steps": 2500,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "bf16",
    "base_model_precision": "fp8-quanto",
    "optimizer": "adamw8bit",
    "save_every_n_steps": 500,
    "sample_every_n_steps": 500,
    "resolution": [512, 768],
    "caption_dropout_rate": 0.05,
    "seed": 42,
    "guidance_scale": 3.5,
    "sample_steps": 4,
    "ema_decay": 0.99,
}

SAMPLE_PROMPTS = [
    "[trigger] a photorealistic portrait of the character, soft studio lighting, 8k",
    "[trigger] the character standing in a rainy forest, cinematic, ultra detailed",
    "[trigger] close-up face, natural daylight, sharp focus, professional photography",
]


def build_yaml_config(
    run_name: str,
    dataset_path: str,
    output_dir: str,
    trigger_word: str,
    **overrides,
) -> str:
    """Return a YAML configuration string for ai-toolkit LoRA training."""
    cfg = {**DEFAULTS, **overrides}
    resolutions = "\n".join(f"            - {r}" for r in cfg["resolution"])
    sample_prompts_yaml = "\n".join(
        f'          - "{p.replace("[trigger]", trigger_word)}"'
        for p in SAMPLE_PROMPTS
    )

    yaml_content = textwrap.dedent(f"""
        job: extension
        config:
          name: {run_name}
          process:
            - type: sd_trainer
              training_folder: "{output_dir}/{run_name}"
              device: cuda:0

              network:
                type: lora
                linear: {cfg['lora_rank']}
                linear_alpha: {cfg['lora_alpha']}

              model:
                name_or_path: "{cfg['model_name']}"
                is_flux: true
                quantize: true
                quantize_type: "{cfg['base_model_precision']}"

              datasets:
                - folder_path: "{dataset_path}"
                  caption_ext: "txt"
                  caption_dropout_rate: {cfg['caption_dropout_rate']}
                  shuffle_tokens: false
                  cache_latents_to_disk: true
                  resolution:
        {resolutions}

              train:
                batch_size: {cfg['batch_size']}
                steps: {cfg['max_train_steps']}
                gradient_accumulation_steps: {cfg['gradient_accumulation_steps']}
                train_unet: true
                train_text_encoder: false
                gradient_checkpointing: true
                noise_scheduler: "flowmatch"
                optimizer: "{cfg['optimizer']}"
                lr: {cfg['learning_rate']}
                lr_scheduler: {cfg['lr_scheduler']}
                lr_warmup_steps: {cfg['warmup_steps']}
                ema_config:
                  use_ema: true
                  ema_decay: {cfg['ema_decay']}
                dtype: {cfg['mixed_precision']}
                skip_first_sample: true

              save:
                dtype: float16
                save_every: {cfg['save_every_n_steps']}
                max_step_saves_to_keep: 4

              sample:
                sampler: "flowmatch"
                sample_every: {cfg['sample_every_n_steps']}
                width: 768
                height: 768
                prompts:
        {sample_prompts_yaml}
                neg: ""
                seed: {cfg['seed']}
                walk_seed: true
                guidance_scale: {cfg['guidance_scale']}
                sample_steps: {cfg['sample_steps']}
    """).lstrip()

    return yaml_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AI-Toolkit YAML config and optionally launch LoRA training."
    )
    parser.add_argument(
        "--run_name", default="flux_consistency_lora", help="Name for this training run"
    )
    parser.add_argument(
        "--dataset_path",
        default="data/character_dataset",
        help="Path to folder containing training images + .txt captions",
    )
    parser.add_argument(
        "--output_dir", default="output", help="Directory to save LoRA checkpoints"
    )
    parser.add_argument(
        "--trigger_word", default="ohwx person", help="Trigger word used in captions"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to save the generated YAML config (default: output/<run_name>.yaml)",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="If set, launch training via ai-toolkit after generating the config",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override max_train_steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--rank", type=int, default=None, help="Override lora_rank")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.steps is not None:
        overrides["max_train_steps"] = args.steps
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.rank is not None:
        overrides["lora_rank"] = args.rank
        overrides["lora_alpha"] = args.rank  # keep alpha == rank

    yaml_str = build_yaml_config(
        run_name=args.run_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        trigger_word=args.trigger_word,
        **overrides,
    )

    config_path = args.config or os.path.join(args.output_dir, f"{args.run_name}.yaml")
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(yaml_str)

    print(f"[train_lora] Config written to: {config_path}")
    print("-" * 60)
    print(yaml_str)
    print("-" * 60)

    if args.launch:
        cmd = [sys.executable, "-m", "toolkit.run", config_path]
        print(f"[train_lora] Launching: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    else:
        print(
            "[train_lora] Config saved. Run training with:\n"
            f"  python -m toolkit.run {config_path}\n"
            "or pass --launch to start automatically."
        )


if __name__ == "__main__":
    main()

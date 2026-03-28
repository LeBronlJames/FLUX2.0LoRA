# LoRA Training Configuration (24 GB VRAM Edition)

> Target hardware: NVIDIA RTX 3090 / 4090 (24 GB VRAM)  
> Base model: `black-forest-labs/FLUX.1-schnell` (klein 9B distilled)  
> Training framework: [Ostris AI-Toolkit](https://github.com/ostris/ai-toolkit)

---

## Full YAML Config

Save this file as `train_flux_lora.yaml` and pass it to `ai-toolkit`:

```yaml
job: extension
config:
  name: flux_consistency_lora
  process:
    - type: sd_trainer
      training_folder: "output/flux_consistency_lora"
      device: cuda:0

      # ── Network ──────────────────────────────────────────────────
      network:
        type: lora
        linear: 16          # rank
        linear_alpha: 16    # alpha

      # ── Base Model ───────────────────────────────────────────────
      model:
        name_or_path: "black-forest-labs/FLUX.1-schnell"
        is_flux: true
        quantize: true                 # FP8 weights
        quantize_type: "fp8-quanto"

      # ── Dataset ──────────────────────────────────────────────────
      datasets:
        - folder_path: "data/character_dataset"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution:
            - 512
            - 768

      # ── Training Hyperparameters ─────────────────────────────────
      train:
        batch_size: 1
        steps: 2500                    # 2000–3000 recommended
        gradient_accumulation_steps: 4
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 5.0e-5
        lr_scheduler: cosine
        lr_warmup_steps: 100
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
        skip_first_sample: true

      # ── Saving ───────────────────────────────────────────────────
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 4

      # ── Sampling (validation) ─────────────────────────────────────
      sample:
        sampler: "flowmatch"
        sample_every: 500
        width: 768
        height: 768
        prompts:
          - "[trigger] a photorealistic portrait of the character, soft studio lighting, 8k"
          - "[trigger] the character standing in a rainy forest, cinematic, ultra detailed"
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 3.5
        sample_steps: 4
```

---

## Parameter Reference

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `rank` / `linear` | 16 | Good balance of expressiveness vs. VRAM |
| `alpha` | 16 | Equal to rank → no effective scaling |
| `learning_rate` | 5e-5 | Stable for FLUX distilled; reduce to 3e-5 if loss spikes |
| `lr_scheduler` | cosine | Smooth decay, prevents late-stage overfitting |
| `warmup_steps` | 100 | Stabilises early training with FP8 base |
| `steps` | 2000–3000 | ~2500 optimal; watch eval sample quality |
| `batch_size` | 1 | 24 GB limit; effective batch = 4 with grad accumulation |
| `gradient_accumulation` | 4 | Simulates batch size 4 |
| `base precision` | FP8 (fp8-quanto) | Reduces base model VRAM by ~40% |
| `optimizer` | AdamW8bit | Further VRAM savings without quality loss |
| `mixed_precision` | BF16 | Numerically stable for FLUX |
| `noise_scheduler` | flowmatch | Required for FLUX.2 family |
| `train_text_encoder` | false | LoRA targets UNet/transformer only |

---

## VRAM Budget Estimate (RTX 4090, 24 GB)

| Component | VRAM |
|-----------|------|
| FLUX.2 klein 9B (FP8) | ~10 GB |
| Optimizer states (AdamW8bit) | ~3 GB |
| Gradients + activations | ~6 GB |
| Latent cache (512px batch=1) | ~1 GB |
| **Total** | **~20 GB** (4 GB headroom) |

---

## Tips

- **Trigger word**: Use a rare token, e.g., `ohwx` or your character name. Add it to every training caption.
- **Overfitting**: If eval samples lose diversity after step 2000, stop early.
- **Underfitting**: Increase steps to 3000 or lower LR to 3e-5.
- **Face sharpness**: Post-process outputs with [CodeFormer](https://github.com/sczhou/CodeFormer) at weight 0.7.

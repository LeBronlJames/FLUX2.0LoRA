# LoRA 训练详细配置 (24GB 显存版本)

本文档记录了在单张 NVIDIA RTX 3090/4090 (24GB VRAM) 上对 **FLUX.2 [klein] 9B Base** 进行 Consistency LoRA 微调的完整配置方案。

---

## 1. 硬件环境

| 配置项 | 规格 |
|--------|------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| CPU | Intel Core i9-13900K |
| RAM | 64GB DDR5 |
| 存储 | 2TB NVMe SSD |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.1 |
| cuDNN | 8.9.x |

---

## 2. 软件环境

| 组件 | 版本 |
|------|------|
| Python | 3.10.12 |
| PyTorch | 2.1.2+cu121 |
| Diffusers | 0.27.0 |
| Transformers | 4.38.0 |
| Accelerate | 0.27.0 |
| ai-toolkit (Ostris) | 0.1.0 (latest) |
| bitsandbytes | 0.43.0 |
| xformers | 0.0.24 |

---

## 3. 基础模型与 LoRA 配置

```yaml
# ai-toolkit config: consistency_lora_24gb.yaml

job: extension
config:
  name: flux2_consistency_lora_v1
  process:
    - type: sd_trainer
      training_folder: "./training_data"
      device: cuda:0
      
      # ─── 模型配置 ───
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true           # FP8 量化，节省显存
        
      # ─── LoRA 配置 ───
      network:
        type: lora
        linear: 16               # rank = 16
        linear_alpha: 16         # alpha = 16
        # 目标模块：transformer 注意力层 + feed-forward
        target_modules:
          - "attn.to_q"
          - "attn.to_k"
          - "attn.to_v"
          - "attn.to_out.0"
          - "ff.net.0.proj"
          - "ff.net.2"
          
      # ─── 训练超参数 ───
      train:
        batch_size: 1
        steps: 2500
        gradient_accumulation_steps: 4
        train_unet: true
        train_text_encoder: false  # 冻结 text encoder，节省显存
        
        optimizer: "adamw8bit"     # Adam 8-bit，显存优化
        learning_rate: 5.0e-5
        lr_scheduler: "cosine"
        lr_warmup_steps: 100
        
        noise_scheduler: "flow_match"
        timestep_sampling: "sigmoid"
        
        # 混合精度
        dtype: bf16
        train_dtype: fp8           # 训练时 FP8 量化
        
        # 梯度优化
        max_grad_norm: 1.0
        gradient_checkpointing: true
        
      # ─── 数据集配置 ───
      datasets:
        - folder_path: "./training_data/art_real_pairs"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true   # 缓存 latent，加速训练
          resolution:
            - [512, 512]
            - [768, 512]
            - [512, 768]
            
      # ─── 保存配置 ───
      save:
        dtype: float16
        save_every: 500
        max_step_saves_to_keep: 4
        push_to_hub: false
        
      # ─── 采样配置（训练中验证）───
      sample:
        sampler: "flowmatch"
        sample_every: 500
        width: 1024
        height: 1024
        prompts:
          - "photorealistic portrait of a woman, cinematic lighting, 8k, ultra-detailed"
          - "photorealistic full body shot, forest background, natural lighting"
        neg: "anime, cartoon, painting, illustration, low quality, blurry"
        seed: 42
        walk_seed: true
        guidance_scale: 3.5
        sample_steps: 28
```

---

## 4. 显存使用估算

| 组件 | 显存占用 (估算) |
|------|------|
| FLUX.2 9B (FP8 量化) | ~9.5 GB |
| LoRA 参数 (rank=16) | ~0.3 GB |
| Optimizer 状态 (Adam 8bit) | ~2.5 GB |
| 激活值 (gradient checkpointing) | ~4.0 GB |
| Latent 缓存 | ~1.5 GB |
| 其他 (框架、CUDA context) | ~1.5 GB |
| **总计** | **~19.3 GB** |

> ✅ 在 RTX 4090 (24GB) 上有约 4.7GB 余量，训练稳定无 OOM。

---

## 5. 数据集准备

### 目录结构

```
training_data/
└── art_real_pairs/
    ├── 001_anime_portrait.png        # 艺术风格图（输入）
    ├── 001_anime_portrait.txt        # 对应 caption
    ├── 002_oil_painting_woman.png
    ├── 002_oil_painting_woman.txt
    ├── ...（共约 60 对）
    └── regularization/               # 正则化数据（可选）
        ├── reg_001_real_photo.png
        └── reg_001_real_photo.txt
```

### Caption 格式

```
# 艺术图 caption 示例
a woman standing in a forest, anime style, detailed eyes, long hair

# 写实图 caption 示例（加 trigger token）
MYCHAR photorealistic portrait, a woman standing in a forest, cinematic lighting, 8k photo
```

> 💡 `MYCHAR` 为 trigger token，替换为你自定义的唯一标识符（如角色姓名缩写）。

---

## 6. 训练启动命令

```bash
# 激活虚拟环境
conda activate flux_lora

# 启动训练（ai-toolkit）
python run.py config/consistency_lora_24gb.yaml

# 或使用自定义脚本
cd scripts/
python train_lora.py \
    --config_path ../config/consistency_lora_24gb.yaml \
    --output_dir ../checkpoints/ \
    --logging_dir ../logs/ \
    --resume_from_checkpoint latest  # 支持断点续训
```

---

## 7. 训练监控

```bash
# TensorBoard 监控
tensorboard --logdir ./logs/ --port 6006

# 或使用 wandb（需 wandb login）
# 在 config 中添加：
# logging:
#   log_every: 10
#   use_wandb: true
#   wandb_project: "flux2-consistency-lora"
```

---

## 8. 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|------|
| CUDA OOM | 显存不足 | 降低 `batch_size=1`，启用 `gradient_checkpointing`，使用 `fp8` 量化 |
| 训练 loss 不降 | 学习率过高/数据质量差 | 调低 LR 至 `1e-5`，检查 caption 格式 |
| 生成图像模糊 | LoRA rank 不足 | 尝试 `rank=32`（需更多显存） |
| 身份一致性差 | 训练步数不足 | 增加至 3000-4000 steps，检查正则化数据 |
| 风格污染残留 | 负向 prompt 不够强 | 增大 `guidance_scale` 至 5.0-7.0 |

---

## 9. 推荐训练调度

```
Steps 0-100:    Warmup（LR 从 0 线性增加至 5e-5）
Steps 100-2000: 主训练（cosine 衰减）
Steps 2000-2500: 精细化（LR 降至 1e-5，专注一致性）
```

建议在 step 500 / 1000 / 1500 / 2000 各保存一个 checkpoint，取主观评测最好的版本作为最终模型。

# FLUX.2 [klein] 9B — Art2Real Narrative Consistency LoRA  
**Identity-consistent Art2Real conversion + multi-scene photoreal storyboard generation.**  
**面向艺术→写实（Art2Real）的身份一致性 LoRA：支持跨场景叙事系列生成。**

[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview / 项目简介
This repository documents a **Consistency LoRA** trained on **FLUX.2 [klein] 9B Base**, targeting:
- **Art → Photoreal (Art2Real)** conversion with reduced identity drift  
- **Cross-scene identity consistency** for narrative/storyboard generation  
- Compatibility with common pipelines (e.g., inpaint/outpaint-based workflows)

---

## Results / 效果展示（请替换为真实结果）
### Art2Real before/after
Place your comparison image at:

![before_after_comparison](assets/before_after_comparison.png)

### Narrative storyboard (3-shot example)
Example storyboard (generic & safe for public repos):  
**Forest walk after rain → Desert camping under the stars → Tropical beach sunset lounge**

Put your storyboard frames at:
- `assets/storyboard_01_forest_after_rain_walk.png`
- `assets/storyboard_02_desert_camping_starry_sky.png`
- `assets/storyboard_03_tropical_beach_sunset_lounge.png`

![storyboard_01](assets/storyboard_01_forest_after_rain_walk.png)
![storyboard_02](assets/storyboard_02_desert_camping_starry_sky.png)
![storyboard_03](assets/storyboard_03_tropical_beach_sunset_lounge.png)

---

## What this solves / 解决什么问题
### Common failure modes in Art2Real
- **Identity drift** across seeds/scenes (face structure, age cues, key identity features)  
  **身份漂移**：跨种子/跨场景五官结构、年龄特征、辨识度不稳定
- **Inconsistency** across a multi-shot narrative series  
  **叙事不一致**：故事板中同一角色像“不同人”
- **Residual stylization** that hurts photoreal texture stability  
  **风格残留**：写实材质/纹理不稳定、仍偏插画感

### Core idea / 核心思路
Train a LoRA that emphasizes an **identity-consistent mapping into the photoreal domain**, so the same character remains recognizable across shots while the scene changes.

训练侧强调“身份一致性映射到写实域”，推理侧通过稳定的 LoRA 权重与身份锚定提示词，配合必要的局部修复（inpaint）与画面扩展（outpaint），实现跨场景的一致叙事输出。

---

## Model Files / 权重文件放置
- Put LoRA weights under: `models/loras/`
- See: `models/loras/README.md` for expected filenames and notes.

---

## Minimal Usage / 最小使用说明
This repo does not enforce a specific UI. Recommended usage pattern:

1) Load **FLUX.2 [klein] 9B Base** in your preferred pipeline/UI  
2) Load the **Consistency LoRA** (start weight `0.8 ~ 1.2`)  
3) Generate a **photoreal anchor frame** from the art input  
4) For hard cases, combine with:
   - **Inpaint**: mask-guided identity correction (face/body)
   - **Outpaint**: extend canvas and create new shots while keeping identity stable

Practical tips / 实战建议：
- Keep identity tokens/prompt template stable across shots  
- Change only scene tokens (location/time/weather/camera) between storyboard frames  
- Fix seed/CFG schedule during comparisons to isolate the LoRA effect

---

## License
MIT. See [LICENSE](LICENSE).

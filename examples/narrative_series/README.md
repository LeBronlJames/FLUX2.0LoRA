# Narrative Series Examples

This directory contains multi-scene narrative storyboard outputs generated
using the Outpaint pipeline with Consistency LoRA.

## Storyboard Series

### Series 1: Forest → Desert → Beach

| File | Scene | Description |
|------|-------|-------------|
| `scene_01_forest_rain.png` | Scene 1 | Woman walking through misty forest after rain |
| `scene_02_desert_starry.png` | Scene 2 | Same woman lying on car roof in desert, starry night |
| `scene_03_tropical_beach.png` | Scene 3 | Same woman relaxing on tropical beach at sunset |

### Series 2: Skyline → Mountain → Ocean

| File | Scene | Description |
|------|-------|-------------|
| `scene_04_urban_rooftop.png` | Scene 1 | Woman on skyscraper rooftop, city panorama at night |
| `scene_05_mountain_summit.png` | Scene 2 | Same woman on snowy mountain summit at dawn |

## Consistency Evaluation

Each series demonstrates the character identity consistency maintained
across completely different environments and lighting conditions through
the Consistency LoRA mechanism.

Key consistency attributes preserved across scenes:
- Facial features (face shape, eye color, skin tone)
- Hair style and color
- Body proportions
- Overall character identity

## Generation Parameters

- **LoRA weight**: 0.85 (slightly reduced for better scene adaptation)
- **Denoise strength**: 0.65 (preserve identity while adapting environment)
- **Guidance scale**: 3.5
- **Steps**: 28
- **Seed**: Sequential (base_seed + scene_index for reproducibility)

See [../../workflows/outpaint_narrative.json](../../workflows/outpaint_narrative.json)
for the ComfyUI workflow used to generate these series.

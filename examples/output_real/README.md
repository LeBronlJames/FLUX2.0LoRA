# Output Real Examples

This directory contains photorealistic conversion results from the
Art-to-Photorealistic pipeline.

## Placeholder Files

Generated results will be saved here with the following naming convention:

| File | Description |
|------|-------------|
| `before_after_comparison_output.png` | Photorealistic output used for README comparison |
| `result_anime_portrait.png` | Converted anime → photorealistic portrait |
| `result_oil_painting.png` | Converted oil painting → photorealistic |
| `result_sketch.png` | Converted sketch → photorealistic |
| `result_inpainted_face.png` | Inpaint result (face region replaced) |
| `result_inpainted_body.png` | Inpaint result (full body region replaced) |

## Quality Metrics

Output images are evaluated on:
- **ArcFace Similarity**: Identity consistency vs. input art character (target ≥ 0.80)
- **CLIP Score**: Photorealistic quality alignment with prompt (target ≥ 0.85)
- **FID**: Distribution distance from real photos (target ≤ 60)

See [../../docs/training_config.md](../../docs/training_config.md) for full evaluation details.

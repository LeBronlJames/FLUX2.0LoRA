# Input Art Examples

This directory contains artistic-style input images used to demonstrate
the Art-to-Photorealistic conversion pipeline.

## Placeholder Files

Place your artistic input images here with the following naming convention:

| File | Description |
|------|-------------|
| `before_after_comparison_input.png` | Artistic input used for the README before/after comparison |
| `sample_anime.png` | Sample anime-style portrait input |
| `sample_oil_painting.png` | Sample oil painting input |
| `sample_sketch.png` | Sample pencil sketch input |
| `sample_watercolor.png` | Sample watercolor input |
| `mask_face.png` | Example face mask for Inpaint workflow (grayscale, white = replace region) |

## Supported Input Styles

- Anime / Manga
- Oil Painting (Renaissance, Impressionist, Baroque)
- Pencil Sketch / Charcoal Drawing
- Watercolor / Gouache
- Digital Art / Concept Art

## Recommended Resolution

Input images should be at least **512×512** pixels. The pipeline will
automatically resize to **1024×1024** for optimal FLUX.2 generation quality.

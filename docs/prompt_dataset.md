# Training Prompt Dataset (~60 Prompts)

> **Trigger word placeholder**: replace `[trigger]` with your chosen token (e.g. `ohwx person`).  
> All prompts follow the structure: `[trigger] [subject description] [scene/environment] [lighting] [quality tags]`

---

## Category 1: Studio / Portrait (12 prompts)

```
[trigger] a photorealistic portrait of a young woman, neutral background, soft box lighting, 8k, ultra detailed
[trigger] close-up face portrait, sharp focus, diffused studio light, skin pores visible, professional photography
[trigger] upper body portrait, white backdrop, rim lighting, high contrast, editorial magazine style
[trigger] headshot, natural daylight from window, catchlights in eyes, shallow depth of field, 85mm lens
[trigger] frontal portrait, gradient grey background, Rembrandt lighting, film grain, cinematic
[trigger] three-quarter profile portrait, warm golden hour light, bokeh background, lifestyle photography
[trigger] black and white portrait, high-contrast Ansel Adams style, dramatic shadows, 4k
[trigger] beauty shot, pastel background, soft diffused light, dewy skin, fashion editorial
[trigger] environmental portrait, blurred urban background, natural light, Canon 5D look
[trigger] portrait with wind-blown hair, outdoor overcast sky, moody atmosphere, analogue film look
[trigger] symmetrical face close-up, harsh direct flash, editorial look, high saturation
[trigger] low-key dramatic portrait, single side-light, deep shadows, noir photography
```

---

## Category 2: Outdoor Natural Environments (12 prompts)

```
[trigger] standing in a rainy forest, wet leaves, misty atmosphere, cinematic, ultra detailed
[trigger] walking along a sun-drenched beach, turquoise water, golden hour, lifestyle photography
[trigger] seated on a desert dune at starry night, Milky Way overhead, long exposure look
[trigger] running through autumn maple forest, falling leaves, motion blur, vibrant colours
[trigger] standing on a mountain cliff at sunrise, dramatic clouds, backlit silhouette
[trigger] wading in a shallow tropical stream, lush jungle, dappled light, paradise aesthetic
[trigger] full body in snow-covered pine forest, breath visible, winter morning, cinematic cold tones
[trigger] sitting under a cherry blossom tree, petals falling, soft spring light, dreamy
[trigger] walking across a lavender field, golden sunset, wide angle, lifestyle
[trigger] at the edge of a waterfall, mist, rainbow, adventure photography, ultra wide
[trigger] night scene in a bamboo grove, moonlight filtering, fog, East Asian aesthetic
[trigger] lying in a grass meadow, flowers around, looking up at blue sky, high saturation
```

---

## Category 3: Urban / Architectural (10 prompts)

```
[trigger] walking on a neon-lit Tokyo street at night, rain reflections, cyberpunk atmosphere
[trigger] standing on a rooftop overlooking city skyline at dusk, golden sky, lifestyle
[trigger] in a narrow European alley, cobblestones, warm cafe lights, street photography
[trigger] on a subway platform, motion blur of passing train, documentary style
[trigger] inside a grand library, warm amber light, bokeh bookshelves, academic aesthetic
[trigger] in front of a graffiti wall, urban streetwear, natural light, fashion editorial
[trigger] at a rooftop pool with infinity edge, Dubai skyline, luxury lifestyle
[trigger] walking through a minimalist white architecture interior, dramatic shadows, fashion
[trigger] in an underground rave venue, colorful strobe lights, atmospheric, editorial
[trigger] on a fire escape in New York, brick wall, golden hour, gritty cinematic
```

---

## Category 4: Artistic Style Cross-Training (8 prompts)

> These prompts help the model learn style-agnostic identity anchoring.

```
[trigger] anime style illustration, soft cel shading, vibrant colours, detailed face
[trigger] oil painting portrait, impressionist brushstrokes, warm palette, museum quality
[trigger] pencil sketch portrait, cross-hatching, detailed shading, fine art
[trigger] watercolour portrait, loose brushwork, pastel tones, artistic
[trigger] comic book style portrait, bold outlines, flat colours, Marvel aesthetic
[trigger] charcoal portrait, dramatic tonal range, fine art, A4 paper texture
[trigger] digital painting, concept art style, detailed environment, ArtStation quality
[trigger] stained glass style portrait, vibrant geometric colour blocks, decorative
```

---

## Category 5: Clothing & Lifestyle (10 prompts)

```
[trigger] wearing a white summer dress, beach boardwalk, golden hour, lifestyle photography
[trigger] in casual streetwear, sitting on stairs, urban environment, natural light
[trigger] in formal evening gown, ballroom, crystal chandelier, luxury editorial
[trigger] in athletic wear, gym environment, sports photography, dynamic pose
[trigger] in a leather jacket, motorcycle in background, dramatic backlight, cinematic
[trigger] in swimwear at a tropical pool, vibrant colours, resort lifestyle photography
[trigger] in traditional Chinese qipao dress, garden setting, soft light, cultural editorial
[trigger] in business suit, office interior, professional headshot style, corporate
[trigger] in cozy knit sweater, autumn cafe window seat, warm tones, lifestyle
[trigger] in military-style jacket, urban rooftop, overcast sky, editorial fashion
```

---

## Category 6: Emotion & Expression (8 prompts)

```
[trigger] laughing joyfully, candid moment, natural light, lifestyle photography
[trigger] contemplative expression, looking into distance, moody light, cinematic
[trigger] surprised expression, wide eyes, dramatic backlight, fashion editorial
[trigger] serene meditative expression, yoga pose, dawn light, wellness photography
[trigger] confident power pose, corporate environment, strong directional light
[trigger] melancholic expression, rainy window, soft overcast light, emotional portrait
[trigger] playful smiling, colourful confetti, birthday celebration, vibrant
[trigger] intense focused expression, dramatic shadows, low key lighting, strong character
```

---

## Total: 60 Prompts

### Caption File Convention

Each image in `data/character_dataset/` should have a corresponding `.txt` file with the same base name:

```
data/character_dataset/
├── img_001.jpg
├── img_001.txt    ← "[trigger] a photorealistic portrait ..."
├── img_002.jpg
├── img_002.txt
...
```

### Recommended Caption Quality Checklist

- [x] Every caption starts with the trigger word
- [x] Subject description is specific (age, gender, hair, etc.)
- [x] Scene/environment is described (1–2 words minimum)
- [x] Lighting condition is mentioned
- [x] At least one quality tag: `8k`, `ultra detailed`, `cinematic`, `professional photography`
- [x] No negative language (save negatives for inference)
